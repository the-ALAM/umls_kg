"""Dagster asset for computing graph metrics (centrality, hierarchy depth)."""

from dagster import asset, AssetExecutionContext
import polars as pl
from typing import Dict, Any

from src.features.centrality import (
    build_graph_from_relationships,
    calculate_eigenvector_centrality,
    calculate_hierarchy_depth,
    calculate_synonym_count,
)
from src.ingestion.loader import load_concept_synonym
from pathlib import Path


@asset(deps=["umls_raw_load"])
def graph_metrics_calc(
    context: AssetExecutionContext,
    umls_raw_load: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Calculate graph metrics: eigenvector centrality and hierarchy depth.
    
    Returns:
        Dictionary with centrality and depth mappings
    """
    context.log.info("Building graph from relationships...")
    
    df_concept = umls_raw_load["concept"]
    df_relationship = umls_raw_load["concept_relationship"]
    
    # Add weight column to relationships (will be used in graph)
    df_relationship_with_weight = df_relationship.with_columns([
        pl.lit(1.0).alias("weight")  # Default weight, can be enhanced with relationship mapper
    ])
    
    # Build graph
    G = build_graph_from_relationships(df_relationship_with_weight, df_concept)
    context.log.info(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Calculate eigenvector centrality
    context.log.info("Calculating eigenvector centrality...")
    centrality = calculate_eigenvector_centrality(G)
    context.log.info(f"Calculated centrality for {len(centrality)} concepts")
    
    # Calculate hierarchy depth
    context.log.info("Calculating hierarchy depth...")
    df_ancestor = umls_raw_load["concept_ancestor"]
    concept_ids = df_concept["concept_id"].to_list()
    depth = calculate_hierarchy_depth(df_ancestor, concept_ids)
    context.log.info(f"Calculated depth for {len(depth)} concepts")
    
    # Calculate synonym count
    context.log.info("Calculating synonym counts...")
    data_dir = Path("data")
    df_synonym = load_concept_synonym(data_dir / "CONCEPT_SYNONYM.csv").collect()
    synonym_count = calculate_synonym_count(df_synonym, concept_ids)
    context.log.info(f"Calculated synonym counts for {len(synonym_count)} concepts")
    
    return {
        "centrality": centrality,
        "hierarchy_depth": depth,
        "synonym_count": synonym_count,
    }

