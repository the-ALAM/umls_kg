"""Dagster asset to compute and cache relevance scores."""

from dagster import asset, AssetExecutionContext
from typing import Dict, Any, Set, List
import numpy as np
from datetime import datetime

from src.scoring.relevance import (
    RelevanceScorer,
    build_relationship_map,
    build_ancestor_map,
)
from src.database.surreal_client import SurrealDBSync


@asset(deps=["surreal_upsert_metrics", "umls_raw_load"])
def relevance_scores_calc(
    context: AssetExecutionContext,
    umls_raw_load: Dict[str, Any],
    graph_metrics_calc: Dict[str, Any],
    concept_embedding_gen: Dict[str, Any],
    authority_scores_calc: Dict[int, float],
) -> Dict[str, Any]:
    """
    Compute relevance scores for concepts against domain clusters.
    
    This is a simplified version that computes scores for concepts within their own domain.
    In production, this would compute scores against specific domain clusters (e.g., DRG codes).
    
    Returns:
        Dictionary with relevance scores
    """
    context.log.info("Computing relevance scores...")
    
    scorer = RelevanceScorer(alpha=0.4, beta=0.3, gamma=0.2, delta=0.1)
    
    # Get data
    df_concept = umls_raw_load["concept"]
    df_relationship = umls_raw_load["concept_relationship"]
    df_ancestor = umls_raw_load["concept_ancestor"]
    
    # Build maps
    relationship_map = build_relationship_map(df_relationship.to_dicts())
    ancestor_map = build_ancestor_map(df_ancestor.to_dicts())
    
    # Get embeddings
    embeddings_dict = {
        concept_id: np.array(embedding)
        for concept_id, embedding in concept_embedding_gen["embeddings"].items()
    }
    
    # Group concepts by domain
    domain_clusters: Dict[str, Set[int]] = {}
    for row in df_concept.iter_rows(named=True):
        domain_id = row["domain_id"]
        concept_id = row["concept_id"]
        
        if domain_id not in domain_clusters:
            domain_clusters[domain_id] = set()
        domain_clusters[domain_id].add(concept_id)
    
    context.log.info(f"Found {len(domain_clusters)} domain clusters")
    
    # Calculate scores for each concept within its domain
    relevance_scores = {}
    
    for domain_id, cluster in domain_clusters.items():
        context.log.info(f"Processing domain {domain_id} with {len(cluster)} concepts...")
        
        # Calculate domain centroid
        domain_centroid = scorer.calculate_domain_cluster_centroid(cluster, embeddings_dict)
        
        # Calculate scores for each concept in the domain
        for concept_id in cluster:
            if concept_id not in embeddings_dict:
                continue
            
            # Build graph paths (simplified - in production, use actual graph traversal)
            graph_paths = {concept_id: {}}  # Placeholder
            
            score_result = scorer.calculate_relevance(
                concept_id=concept_id,
                domain_cluster=cluster,
                relationship_map=relationship_map,
                ancestor_map=ancestor_map,
                concept_embedding=embeddings_dict[concept_id],
                domain_cluster_centroid=domain_centroid,
                graph_paths=graph_paths,
                source_authority=authority_scores_calc.get(concept_id, 0.5),
            )
            
            relevance_scores[concept_id] = {
                "domain_cluster_id": domain_id,
                **score_result,
            }
    
    context.log.info(f"Computed {len(relevance_scores)} relevance scores")
    
    return {
        "scores": relevance_scores,
        "timestamp": datetime.now().isoformat(),
    }


@asset(deps=["relevance_scores_calc"])
def surreal_upsert_relevance_scores(
    context: AssetExecutionContext,
    relevance_scores_calc: Dict[str, Any],
) -> None:
    """Upsert relevance scores to SurrealDB."""
    client = SurrealDBSync()
    
    try:
        client.connect()
        context.log.info("Connected to SurrealDB")
        
        scores = relevance_scores_calc["scores"]
        context.log.info(f"Upserting {len(scores)} relevance scores...")
        
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
                    "timestamp": relevance_scores_calc["timestamp"],
                }
                
                record_id = f"{concept_id}_{score_data['domain_cluster_id']}"
                client.upsert("relevance_score", record, record_id)
            
            if (i // batch_size) % 10 == 0:
                context.log.info(f"Processed {i + len(batch)} scores")
        
        context.log.info("Completed relevance score upsert")
    finally:
        client.close()

