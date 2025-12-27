"""Dagster asset for calculating vocabulary authority scores."""

from dagster import asset, AssetExecutionContext
import polars as pl
from typing import Dict, Any

from src.features.authority import get_vocabulary_authority


@asset(deps=["umls_raw_load"])
def authority_scores_calc(
    context: AssetExecutionContext,
    umls_raw_load: Dict[str, Any],
) -> Dict[int, float]:
    """
    Calculate vocabulary authority scores for all concepts.
    
    Returns:
        Dictionary mapping concept_id to authority score
    """
    context.log.info("Calculating vocabulary authority scores...")
    
    df_concept = umls_raw_load["concept"]
    
    # Calculate authority for each concept
    authority_map = {}
    for row in df_concept.iter_rows(named=True):
        concept_id = row["concept_id"]
        vocabulary_id = row["vocabulary_id"]
        authority = get_vocabulary_authority(vocabulary_id)
        authority_map[concept_id] = authority
    
    context.log.info(f"Calculated authority scores for {len(authority_map)} concepts")
    
    # Log distribution
    authority_values = list(authority_map.values())
    context.log.info(f"Authority score range: {min(authority_values):.2f} - {max(authority_values):.2f}")
    context.log.info(f"Average authority: {sum(authority_values) / len(authority_values):.2f}")
    
    return authority_map

