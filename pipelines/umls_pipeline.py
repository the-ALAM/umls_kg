"""Main Dagster pipeline for UMLS knowledge graph system."""

from dagster import Definitions, load_assets_from_modules

from assets import (
    raw_ingest,
    graph_features,
    semantic_features,
    authority_scores,
    metric_sync,
    relevance_scores,
    audit_report,
)

# Import all assets
all_assets = [
    raw_ingest.umls_raw_load,
    raw_ingest.surreal_ingest_concepts,
    raw_ingest.surreal_ingest_relationships,
    raw_ingest.surreal_ingest_ancestors,
    raw_ingest.surreal_ingest_reference_tables,
    graph_features.graph_metrics_calc,
    semantic_features.concept_embedding_gen,
    authority_scores.authority_scores_calc,
    metric_sync.surreal_upsert_metrics,
    relevance_scores.relevance_scores_calc,
    relevance_scores.surreal_upsert_relevance_scores,
    audit_report.audit_report,
]


defs = Definitions(
    assets=all_assets,
)

