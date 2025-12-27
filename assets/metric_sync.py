"""Dagster asset for syncing metrics to SurrealDB."""

from dagster import asset, AssetExecutionContext
from typing import Dict, Any
import hashlib
from datetime import datetime

from src.database.surreal_client import SurrealDBSync


@asset(
    deps=["graph_metrics_calc", "concept_embedding_gen", "authority_scores_calc"]
)
def surreal_upsert_metrics(
    context: AssetExecutionContext,
    graph_metrics_calc: Dict[str, Any],
    concept_embedding_gen: Dict[str, Any],
    authority_scores_calc: Dict[int, float],
) -> None:
    """
    Upsert computed metrics to SurrealDB metric sidecar table.
    
    Args:
        graph_metrics_calc: Graph metrics (centrality, depth, synonym_count)
        concept_embedding_gen: Embedding data
        authority_scores_calc: Authority scores
    """
    client = SurrealDBSync()
    
    try:
        client.connect()
        context.log.info("Connected to SurrealDB")
        
        # Get all concept IDs from embeddings
        concept_ids = list(concept_embedding_gen["embeddings"].keys())
        context.log.info(f"Upserting metrics for {len(concept_ids)} concepts...")
        
        batch_size = 1000
        for i in range(0, len(concept_ids), batch_size):
            batch_ids = concept_ids[i:i + batch_size]
            
            for concept_id in batch_ids:
                # Gather all metrics
                centrality = graph_metrics_calc["centrality"].get(concept_id, 0.0)
                hierarchy_depth = graph_metrics_calc["hierarchy_depth"].get(concept_id, 0)
                synonym_count = graph_metrics_calc["synonym_count"].get(concept_id, 0)
                semantic_embed = concept_embedding_gen["embeddings"].get(concept_id, [])
                source_authority = authority_scores_calc.get(concept_id, 0.5)
                
                # Generate fingerprint
                fingerprint_data = f"{concept_id}_{centrality}_{hierarchy_depth}_{synonym_count}_{concept_embedding_gen['model_version']}"
                fingerprint = hashlib.sha256(fingerprint_data.encode()).hexdigest()
                
                # Create metric record
                metric_record = {
                    "concept_id": f"concept:{concept_id}",
                    "source_authority": float(source_authority),
                    "centrality": float(centrality),
                    "semantic_embed": semantic_embed,
                    "hierarchy_depth": int(hierarchy_depth),
                    "synonym_count": int(synonym_count),
                    "model_version": concept_embedding_gen["model_version"],
                    "fingerprint": fingerprint,
                    "timestamp": datetime.now().isoformat(),
                }
                
                # Upsert to SurrealDB
                client.upsert("metric", metric_record, str(concept_id))
            
            if (i // batch_size) % 10 == 0:
                context.log.info(f"Processed {i + len(batch_ids)} metrics")
        
        context.log.info("Completed metric upsert")
    finally:
        client.close()

