"""Dagster asset for loading raw UMLS data into SurrealDB (Layer 1)."""

from dagster import asset, AssetExecutionContext
from pathlib import Path
import polars as pl
from typing import Dict, Any

from src.ingestion.loader import (
    load_concept,
    load_concept_relationship,
    load_concept_ancestor,
    load_vocabulary,
    load_domain,
    load_relationship,
)
from src.ingestion.cleaner import (
    clean_concept_names,
    clean_relationships,
    clean_ancestors,
)
from src.database.surreal_client import SurrealDBSync
from src.transformers.relationship_mapper import (
    get_relationship_category,
    get_relationship_weight,
)


@asset
def umls_raw_load(context: AssetExecutionContext) -> Dict[str, Any]:
    """
    Load and clean raw UMLS CSV files using Polars.
    
    Returns:
        Dictionary with DataFrames for each table
    """
    data_dir = Path("data")
    
    context.log.info("Loading CONCEPT.csv...")
    df_concept = clean_concept_names(load_concept(data_dir / "CONCEPT.csv")).collect()
    
    context.log.info("Loading CONCEPT_RELATIONSHIP.csv...")
    df_relationship = clean_relationships(
        load_concept_relationship(data_dir / "CONCEPT_RELATIONSHIP.csv")
    ).collect()
    
    context.log.info("Loading CONCEPT_ANCESTOR.csv...")
    df_ancestor = clean_ancestors(
        load_concept_ancestor(data_dir / "CONCEPT_ANCESTOR.csv")
    ).collect()
    
    context.log.info("Loading VOCABULARY.csv...")
    df_vocabulary = load_vocabulary(data_dir / "VOCABULARY.csv").collect()
    
    context.log.info("Loading DOMAIN.csv...")
    df_domain = load_domain(data_dir / "DOMAIN.csv").collect()
    
    context.log.info("Loading RELATIONSHIP.csv...")
    df_relationship_ref = load_relationship(data_dir / "RELATIONSHIP.csv").collect()
    
    context.log.info(f"Loaded {len(df_concept)} concepts")
    context.log.info(f"Loaded {len(df_relationship)} relationships")
    context.log.info(f"Loaded {len(df_ancestor)} ancestor relationships")
    
    return {
        "concept": df_concept,
        "concept_relationship": df_relationship,
        "concept_ancestor": df_ancestor,
        "vocabulary": df_vocabulary,
        "domain": df_domain,
        "relationship": df_relationship_ref,
    }


@asset(deps=[umls_raw_load])
def surreal_ingest_concepts(context: AssetExecutionContext, umls_raw_load: Dict[str, Any]) -> None:
    """Ingest concepts as nodes into SurrealDB."""
    client = SurrealDBSync()
    
    try:
        client.connect()
        context.log.info("Connected to SurrealDB")
        
        df_concept = umls_raw_load["concept"]
        context.log.info(f"Ingesting {len(df_concept)} concepts...")
        
        # Convert to records and upsert
        records = df_concept.to_dicts()
        batch_size = 1000
        
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            for record in batch:
                concept_id = str(record["concept_id"])
                # Prepare record for SurrealDB
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
                context.log.info(f"Processed {i + len(batch)} concepts")
        
        context.log.info("Completed concept ingestion")
    finally:
        client.close()


@asset(deps=[umls_raw_load])
def surreal_ingest_relationships(context: AssetExecutionContext, umls_raw_load: Dict[str, Any]) -> None:
    """Ingest relationships as edges into SurrealDB."""
    client = SurrealDBSync()
    
    try:
        client.connect()
        context.log.info("Connected to SurrealDB")
        
        df_relationship = umls_raw_load["concept_relationship"]
        context.log.info(f"Ingesting {len(df_relationship)} relationships...")
        
        records = df_relationship.to_dicts()
        batch_size = 1000
        
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            for record in batch:
                concept_id_1 = str(record["concept_id_1"])
                concept_id_2 = str(record["concept_id_2"])
                relationship_id = record["relationship_id"]
                
                # Get category and weight
                category = get_relationship_category(relationship_id)
                weight = get_relationship_weight(relationship_id)
                
                # Create edge record
                edge_record = {
                    "in": f"concept:{concept_id_1}",
                    "out": f"concept:{concept_id_2}",
                    "relationship_id": relationship_id,
                    "category": category,
                    "weight": weight,
                    "valid_start_date": record["valid_start_date"],
                    "valid_end_date": record["valid_end_date"],
                }
                
                # Use a unique edge ID
                edge_id = f"{concept_id_1}_{concept_id_2}_{relationship_id}"
                client.upsert("relates_to", edge_record, edge_id)
            
            if (i // batch_size) % 10 == 0:
                context.log.info(f"Processed {i + len(batch)} relationships")
        
        context.log.info("Completed relationship ingestion")
    finally:
        client.close()


@asset(deps=[umls_raw_load])
def surreal_ingest_ancestors(context: AssetExecutionContext, umls_raw_load: Dict[str, Any]) -> None:
    """Ingest ancestor relationships as edges into SurrealDB."""
    client = SurrealDBSync()
    
    try:
        client.connect()
        context.log.info("Connected to SurrealDB")
        
        df_ancestor = umls_raw_load["concept_ancestor"]
        context.log.info(f"Ingesting {len(df_ancestor)} ancestor relationships...")
        
        records = df_ancestor.to_dicts()
        batch_size = 1000
        
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            for record in batch:
                ancestor_id = str(record["ancestor_concept_id"])
                descendant_id = str(record["descendant_concept_id"])
                
                # Create ancestor edge
                edge_record = {
                    "in": f"concept:{ancestor_id}",
                    "out": f"concept:{descendant_id}",
                    "min_levels_of_separation": int(record["min_levels_of_separation"]),
                    "max_levels_of_separation": int(record["max_levels_of_separation"]),
                }
                
                edge_id = f"{ancestor_id}_{descendant_id}"
                client.upsert("is_ancestor_of", edge_record, edge_id)
            
            if (i // batch_size) % 10 == 0:
                context.log.info(f"Processed {i + len(batch)} ancestor relationships")
        
        context.log.info("Completed ancestor ingestion")
    finally:
        client.close()


@asset(deps=[umls_raw_load])
def surreal_ingest_reference_tables(context: AssetExecutionContext, umls_raw_load: Dict[str, Any]) -> None:
    """Ingest vocabulary, domain, and relationship reference tables."""
    client = SurrealDBSync()
    
    try:
        client.connect()
        context.log.info("Connected to SurrealDB")
        
        # Ingest vocabularies
        df_vocabulary = umls_raw_load["vocabulary"]
        context.log.info(f"Ingesting {len(df_vocabulary)} vocabularies...")
        for record in df_vocabulary.to_dicts():
            vocab_id = record["vocabulary_id"]
            client.upsert("vocabulary", record, vocab_id)
        
        # Ingest domains
        df_domain = umls_raw_load["domain"]
        context.log.info(f"Ingesting {len(df_domain)} domains...")
        for record in df_domain.to_dicts():
            domain_id = record["domain_id"]
            client.upsert("domain", record, domain_id)
        
        # Ingest relationships
        df_relationship = umls_raw_load["relationship"]
        context.log.info(f"Ingesting {len(df_relationship)} relationship definitions...")
        for record in df_relationship.to_dicts():
            rel_id = record["relationship_id"]
            client.upsert("relationship", record, rel_id)
        
        context.log.info("Completed reference table ingestion")
    finally:
        client.close()

