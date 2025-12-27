"""Dagster asset for generating semantic embeddings."""

from dagster import asset, AssetExecutionContext
import polars as pl
import numpy as np
from typing import Dict, Any
import hashlib
from datetime import datetime

from src.ingestion.cleaner import combine_concept_text
from src.features.embeddings import create_embedding_generator


@asset(deps=["umls_raw_load"])
def concept_embedding_gen(
    context: AssetExecutionContext,
    umls_raw_load: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Generate semantic embeddings for concepts using local LLM.
    
    Returns:
        Dictionary with embeddings and metadata
    """
    context.log.info("Preparing concept text for embedding...")
    
    df_concept = umls_raw_load["concept"]
    
    # Load synonyms
    from pathlib import Path
    from src.ingestion.loader import load_concept_synonym
    data_dir = Path("data")
    df_synonym = load_concept_synonym(data_dir / "CONCEPT_SYNONYM.csv").collect()
    
    # Combine concept names with synonyms
    df_with_text = combine_concept_text(df_concept.lazy(), df_synonym.lazy()).collect()
    
    context.log.info(f"Generating embeddings for {len(df_with_text)} concepts...")
    
    # Initialize embedding generator
    generator = create_embedding_generator(
        model_name="BAAI/bge-large-en-v1.5",
        batch_size=10000,
    )
    
    # Extract texts
    texts = df_with_text["concept_text"].to_list()
    concept_ids = df_with_text["concept_id"].to_list()
    
    # Generate embeddings
    embeddings = generator.generate_embeddings(texts, show_progress=True)
    
    context.log.info(f"Generated embeddings with shape {embeddings.shape}")
    
    # Create mapping from concept_id to embedding
    embedding_map = {
        concept_id: embedding.tolist()
        for concept_id, embedding in zip(concept_ids, embeddings)
    }
    
    # Generate model version fingerprint
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

