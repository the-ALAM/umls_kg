"""Standalone script to run UMLS KG pipeline without Dagster."""

from pathlib import Path
from typing import Dict, Any, Set
import polars as pl
import numpy as np
import hashlib
from datetime import datetime

# Step 1: Load and clean data
from src.ingestion.loader import (
    load_concept,
    load_concept_relationship,
    load_concept_ancestor,
    load_vocabulary,
    load_domain,
    load_relationship,
    load_concept_synonym,
)
from src.ingestion.cleaner import (
    clean_concept_names,
    clean_relationships,
    clean_ancestors,
    combine_concept_text,
)
from src.database import SurrealDBSync, DatabaseConnectionError, QueryExecutionError
from src.database.schema import apply_schema
from src.transformers.relationship_mapper import (
    get_relationship_category,
    get_relationship_weight,
)
from src.features.centrality import (
    build_graph_from_relationships,
    calculate_eigenvector_centrality,
    calculate_hierarchy_depth,
    calculate_synonym_count,
)
from src.features.embeddings import create_embedding_generator
from src.features.authority import get_vocabulary_authority
from src.scoring.relevance import (
    RelevanceScorer,
    build_relationship_map,
    build_ancestor_map,
)


def step1_load_data() -> Dict[str, Any]:
    """Step 1: Load and clean raw UMLS CSV files."""
    print("Step 1: Loading and cleaning data...")
    data_dir = Path("data")
    
    print("  Loading CONCEPT.csv...")
    df_concept = clean_concept_names(load_concept(data_dir / "CONCEPT.csv")).collect()
    
    print("  Loading CONCEPT_RELATIONSHIP.csv...")
    df_relationship = clean_relationships(
        load_concept_relationship(data_dir / "CONCEPT_RELATIONSHIP.csv")
    ).collect()
    
    print("  Loading CONCEPT_ANCESTOR.csv...")
    df_ancestor = clean_ancestors(
        load_concept_ancestor(data_dir / "CONCEPT_ANCESTOR.csv")
    ).collect()
    
    print("  Loading VOCABULARY.csv...")
    df_vocabulary = load_vocabulary(data_dir / "VOCABULARY.csv").collect()
    
    print("  Loading DOMAIN.csv...")
    df_domain = load_domain(data_dir / "DOMAIN.csv").collect()
    
    print("  Loading RELATIONSHIP.csv...")
    df_relationship_ref = load_relationship(data_dir / "RELATIONSHIP.csv").collect()
    
    print(f"  ✓ Loaded {len(df_concept)} concepts, {len(df_relationship)} relationships")
    
    return {
        "concept": df_concept,
        "concept_relationship": df_relationship,
        "concept_ancestor": df_ancestor,
        "vocabulary": df_vocabulary,
        "domain": df_domain,
        "relationship": df_relationship_ref,
    }


def step2_setup_database():
    """Step 2: Setup SurrealDB schema."""
    print("\nStep 2: Setting up SurrealDB schema...")
    try:
        with SurrealDBSync() as client:
            apply_schema(client)
            print("  ✓ Schema applied")
    except DatabaseConnectionError as e:
        print(f"  ✗ Database connection failed: {e}")
        raise
    except Exception as e:
        print(f"  ✗ Schema setup failed: {e}")
        raise


def step3_ingest_raw_data(umls_data: Dict[str, Any]):
    """Step 3: Ingest raw data into SurrealDB."""
    print("\nStep 3: Ingesting raw data into SurrealDB...")
    
    try:
        with SurrealDBSync() as client:
            # Ingest concepts
            print("  Ingesting concepts...")
            df_concept = umls_data["concept"]
            records = df_concept.to_dicts()
            batch_size = 1000
            
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                for record in batch:
                    concept_id = str(record["concept_id"])
                    db_record = {
                        "concept_id": int(record["concept_id"]),
                        "name": record["concept_name"],
                        "vocabulary_id": record["vocabulary_id"],
                        "domain_id": record["domain_id"],
                        "concept_code": record["concept_code"],
                        "concept_class_id": record["concept_class_id"],
                        "standard_concept": record["standard_concept"],
                        "valid_start_date": record["valid_start_date"],
                        "valid_end_date": record["valid_end_date"],
                        "invalid_reason": record.get("invalid_reason"),
                    }
                    client.upsert("concept", db_record, concept_id)
                
                if (i // batch_size) % 10 == 0:
                    print(f"    Processed {i + len(batch)} concepts")
            
            print(f"  ✓ Ingested {len(records)} concepts")
            
            # Ingest relationships
            print("  Ingesting relationships...")
            df_relationship = umls_data["concept_relationship"]
            records = df_relationship.to_dicts()
            
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                for record in batch:
                    concept_id_1 = str(record["concept_id_1"])
                    concept_id_2 = str(record["concept_id_2"])
                    relationship_id = record["relationship_id"]
                    
                    category = get_relationship_category(relationship_id)
                    weight = get_relationship_weight(relationship_id)
                    
                    edge_record = {
                        "in": f"concept:{concept_id_1}",
                        "out": f"concept:{concept_id_2}",
                        "relationship_id": relationship_id,
                        "category": category,
                        "weight": weight,
                        "valid_start_date": record["valid_start_date"],
                        "valid_end_date": record["valid_end_date"],
                    }
                    
                    edge_id = f"{concept_id_1}_{concept_id_2}_{relationship_id}"
                    client.upsert("relates_to", edge_record, edge_id)
                
                if (i // batch_size) % 10 == 0:
                    print(f"    Processed {i + len(batch)} relationships")
            
            print(f"  ✓ Ingested {len(records)} relationships")
            
            # Ingest ancestors
            print("  Ingesting ancestors...")
            df_ancestor = umls_data["concept_ancestor"]
            records = df_ancestor.to_dicts()
            
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                for record in batch:
                    ancestor_id = str(record["ancestor_concept_id"])
                    descendant_id = str(record["descendant_concept_id"])
                    
                    edge_record = {
                        "in": f"concept:{ancestor_id}",
                        "out": f"concept:{descendant_id}",
                        "min_levels_of_separation": int(record["min_levels_of_separation"]),
                        "max_levels_of_separation": int(record["max_levels_of_separation"]),
                    }
                    
                    edge_id = f"{ancestor_id}_{descendant_id}"
                    client.upsert("is_ancestor_of", edge_record, edge_id)
            
            print(f"  ✓ Ingested {len(records)} ancestor relationships")
            
            # Ingest reference tables
            print("  Ingesting reference tables...")
            for record in umls_data["vocabulary"].to_dicts():
                client.upsert("vocabulary", record, record["vocabulary_id"])
            for record in umls_data["domain"].to_dicts():
                client.upsert("domain", record, record["domain_id"])
            for record in umls_data["relationship"].to_dicts():
                client.upsert("relationship", record, record["relationship_id"])
            
            print("  ✓ Ingested reference tables")
    except DatabaseConnectionError as e:
        print(f"  ✗ Database connection failed: {e}")
        raise
    except QueryExecutionError as e:
        print(f"  ✗ Query execution failed: {e}")
        raise
    except Exception as e:
        print(f"  ✗ Data ingestion failed: {e}")
        raise


def step4_calculate_graph_metrics(umls_data: Dict[str, Any]) -> Dict[str, Any]:
    """Step 4: Calculate graph metrics."""
    print("\nStep 4: Calculating graph metrics...")
    
    df_concept = umls_data["concept"]
    df_relationship = umls_data["concept_relationship"]
    
    # Add weight column
    df_relationship_with_weight = df_relationship.with_columns([
        pl.lit(1.0).alias("weight")
    ])
    
    # Build graph
    print("  Building graph...")
    G = build_graph_from_relationships(df_relationship_with_weight, df_concept)
    print(f"    Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Calculate centrality
    print("  Calculating eigenvector centrality...")
    centrality = calculate_eigenvector_centrality(G)
    print(f"    ✓ Calculated centrality for {len(centrality)} concepts")
    
    # Calculate depth
    print("  Calculating hierarchy depth...")
    df_ancestor = umls_data["concept_ancestor"]
    concept_ids = df_concept["concept_id"].to_list()
    depth = calculate_hierarchy_depth(df_ancestor, concept_ids)
    print(f"    ✓ Calculated depth for {len(depth)} concepts")
    
    # Calculate synonym count
    print("  Calculating synonym counts...")
    data_dir = Path("data")
    df_synonym = load_concept_synonym(data_dir / "CONCEPT_SYNONYM.csv").collect()
    synonym_count = calculate_synonym_count(df_synonym, concept_ids)
    print(f"    ✓ Calculated synonym counts for {len(synonym_count)} concepts")
    
    return {
        "centrality": centrality,
        "hierarchy_depth": depth,
        "synonym_count": synonym_count,
    }


def step5_generate_embeddings(umls_data: Dict[str, Any]) -> Dict[str, Any]:
    """Step 5: Generate semantic embeddings."""
    print("\nStep 5: Generating semantic embeddings...")
    
    df_concept = umls_data["concept"]
    
    # Load synonyms
    data_dir = Path("data")
    df_synonym = load_concept_synonym(data_dir / "CONCEPT_SYNONYM.csv").collect()
    
    # Combine concept names with synonyms
    df_with_text = combine_concept_text(df_concept.lazy(), df_synonym.lazy()).collect()
    
    print(f"  Generating embeddings for {len(df_with_text)} concepts...")
    
    # Initialize generator
    generator = create_embedding_generator(
        model_name="BAAI/bge-large-en-v1.5",
        batch_size=10000,
    )
    
    # Extract texts and generate embeddings
    texts = df_with_text["concept_text"].to_list()
    concept_ids = df_with_text["concept_id"].to_list()
    
    embeddings = generator.generate_embeddings(texts, show_progress=True)
    print(f"    ✓ Generated embeddings with shape {embeddings.shape}")
    
    # Create mapping
    embedding_map = {
        concept_id: embedding.tolist()
        for concept_id, embedding in zip(concept_ids, embeddings)
    }
    
    # Generate model version
    model_version = hashlib.sha256(
        f"{generator.model_name}_{generator.embedding_dim}_{datetime.now().isoformat()}".encode()
    ).hexdigest()[:16]
    
    return {
        "embeddings": embedding_map,
        "model_name": generator.model_name,
        "embedding_dim": generator.embedding_dim,
        "model_version": model_version,
        "timestamp": datetime.now().isoformat(),
    }


def step6_calculate_authority_scores(umls_data: Dict[str, Any]) -> Dict[int, float]:
    """Step 6: Calculate vocabulary authority scores."""
    print("\nStep 6: Calculating vocabulary authority scores...")
    
    df_concept = umls_data["concept"]
    authority_map = {}
    
    for row in df_concept.iter_rows(named=True):
        concept_id = row["concept_id"]
        vocabulary_id = row["vocabulary_id"]
        authority_map[concept_id] = get_vocabulary_authority(vocabulary_id)
    
    print(f"  ✓ Calculated authority scores for {len(authority_map)} concepts")
    return authority_map


def step7_upsert_metrics(
    graph_metrics: Dict[str, Any],
    embeddings: Dict[str, Any],
    authority_scores: Dict[int, float],
):
    """Step 7: Upsert metrics to SurrealDB."""
    print("\nStep 7: Upserting metrics to SurrealDB...")
    
    try:
        with SurrealDBSync() as client:
            concept_ids = list(embeddings["embeddings"].keys())
            print(f"  Upserting metrics for {len(concept_ids)} concepts...")
            
            batch_size = 1000
            for i in range(0, len(concept_ids), batch_size):
                batch_ids = concept_ids[i:i + batch_size]
                
                for concept_id in batch_ids:
                    centrality = graph_metrics["centrality"].get(concept_id, 0.0)
                    hierarchy_depth = graph_metrics["hierarchy_depth"].get(concept_id, 0)
                    synonym_count = graph_metrics["synonym_count"].get(concept_id, 0)
                    semantic_embed = embeddings["embeddings"].get(concept_id, [])
                    source_authority = authority_scores.get(concept_id, 0.5)
                    
                    fingerprint_data = f"{concept_id}_{centrality}_{hierarchy_depth}_{synonym_count}_{embeddings['model_version']}"
                    fingerprint = hashlib.sha256(fingerprint_data.encode()).hexdigest()
                    
                    metric_record = {
                        "concept_id": f"concept:{concept_id}",
                        "source_authority": float(source_authority),
                        "centrality": float(centrality),
                        "semantic_embed": semantic_embed,
                        "hierarchy_depth": int(hierarchy_depth),
                        "synonym_count": int(synonym_count),
                        "model_version": embeddings["model_version"],
                        "fingerprint": fingerprint,
                        "timestamp": datetime.now().isoformat(),
                    }
                    
                    client.upsert("metric", metric_record, str(concept_id))
                
                if (i // batch_size) % 10 == 0:
                    print(f"    Processed {i + len(batch_ids)} metrics")
            
            print("  ✓ Completed metric upsert")
    except DatabaseConnectionError as e:
        print(f"  ✗ Database connection failed: {e}")
        raise
    except QueryExecutionError as e:
        print(f"  ✗ Query execution failed: {e}")
        raise
    except Exception as e:
        print(f"  ✗ Metric upsert failed: {e}")
        raise


def step8_calculate_relevance_scores(
    umls_data: Dict[str, Any],
    graph_metrics: Dict[str, Any],
    embeddings: Dict[str, Any],
    authority_scores: Dict[int, float],
) -> Dict[str, Any]:
    """Step 8: Calculate relevance scores."""
    print("\nStep 8: Calculating relevance scores...")
    
    scorer = RelevanceScorer(alpha=0.4, beta=0.3, gamma=0.2, delta=0.1)
    
    df_concept = umls_data["concept"]
    df_relationship = umls_data["concept_relationship"]
    df_ancestor = umls_data["concept_ancestor"]
    
    # Build maps
    relationship_map = build_relationship_map(df_relationship.to_dicts())
    ancestor_map = build_ancestor_map(df_ancestor.to_dicts())
    
    # Get embeddings
    embeddings_dict = {
        concept_id: np.array(embedding)
        for concept_id, embedding in embeddings["embeddings"].items()
    }
    
    # Group concepts by domain
    domain_clusters: Dict[str, Set[int]] = {}
    for row in df_concept.iter_rows(named=True):
        domain_id = row["domain_id"]
        concept_id = row["concept_id"]
        
        if domain_id not in domain_clusters:
            domain_clusters[domain_id] = set()
        domain_clusters[domain_id].add(concept_id)
    
    print(f"  Found {len(domain_clusters)} domain clusters")
    
    # Calculate scores
    relevance_scores = {}
    
    for domain_id, cluster in domain_clusters.items():
        print(f"  Processing domain {domain_id} with {len(cluster)} concepts...")
        
        domain_centroid = scorer.calculate_domain_cluster_centroid(cluster, embeddings_dict)
        
        for concept_id in cluster:
            if concept_id not in embeddings_dict:
                continue
            
            graph_paths = {concept_id: {}}  # Placeholder
            
            score_result = scorer.calculate_relevance(
                concept_id=concept_id,
                domain_cluster=cluster,
                relationship_map=relationship_map,
                ancestor_map=ancestor_map,
                concept_embedding=embeddings_dict[concept_id],
                domain_cluster_centroid=domain_centroid,
                graph_paths=graph_paths,
                source_authority=authority_scores.get(concept_id, 0.5),
            )
            
            relevance_scores[concept_id] = {
                "domain_cluster_id": domain_id,
                **score_result,
            }
    
    print(f"  ✓ Computed {len(relevance_scores)} relevance scores")
    
    return {
        "scores": relevance_scores,
        "timestamp": datetime.now().isoformat(),
    }


def step9_upsert_relevance_scores(relevance_scores: Dict[str, Any]):
    """Step 9: Upsert relevance scores to SurrealDB."""
    print("\nStep 9: Upserting relevance scores to SurrealDB...")
    
    try:
        with SurrealDBSync() as client:
            scores = relevance_scores["scores"]
            print(f"  Upserting {len(scores)} relevance scores...")
            
            batch_size = 1000
            score_items = list(scores.items())
            
            for i in range(0, len(score_items), batch_size):
                batch = score_items[i:i + batch_size]
                
                for concept_id, score_data in batch:
                    record = {
                        "concept_id": f"concept:{concept_id}",
                        "domain_cluster_id": score_data["domain_cluster_id"],
                        "score": float(score_data["score"]),
                        "s_struct": float(score_data["s_struct"]),
                        "s_sem": float(score_data["s_sem"]),
                        "s_density": float(score_data["s_density"]),
                        "s_authority": float(score_data["s_authority"]),
                        "timestamp": relevance_scores["timestamp"],
                    }
                    
                    record_id = f"{concept_id}_{score_data['domain_cluster_id']}"
                    client.upsert("relevance_score", record, record_id)
                
                if (i // batch_size) % 10 == 0:
                    print(f"    Processed {i + len(batch)} scores")
            
            print("  ✓ Completed relevance score upsert")
    except DatabaseConnectionError as e:
        print(f"  ✗ Database connection failed: {e}")
        raise
    except QueryExecutionError as e:
        print(f"  ✗ Query execution failed: {e}")
        raise
    except Exception as e:
        print(f"  ✗ Relevance score upsert failed: {e}")
        raise


def main():
    """Run the complete pipeline manually."""
    print("=" * 60)
    print("UMLS Knowledge Graph Pipeline (Manual Execution)")
    print("=" * 60)
    
    # Step 1: Load data
    umls_data = step1_load_data()
    
    # Step 2: Setup database
    step2_setup_database()
    
    # Step 3: Ingest raw data
    step3_ingest_raw_data(umls_data)
    
    # Step 4: Calculate graph metrics
    graph_metrics = step4_calculate_graph_metrics(umls_data)
    
    # Step 5: Generate embeddings
    embeddings = step5_generate_embeddings(umls_data)
    
    # Step 6: Calculate authority scores
    authority_scores = step6_calculate_authority_scores(umls_data)
    
    # Step 7: Upsert metrics
    step7_upsert_metrics(graph_metrics, embeddings, authority_scores)
    
    # Step 8: Calculate relevance scores
    relevance_scores = step8_calculate_relevance_scores(
        umls_data, graph_metrics, embeddings, authority_scores
    )
    
    # Step 9: Upsert relevance scores
    step9_upsert_relevance_scores(relevance_scores)
    
    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
