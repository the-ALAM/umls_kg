"""Data quality checks and validation for UMLS data."""

import polars as pl
from typing import Dict, List, Tuple


def validate_referential_integrity(
    df_relationships: pl.LazyFrame,
    df_concepts: pl.LazyFrame,
) -> Tuple[bool, List[str]]:
    """
    Validate that all concept IDs in relationships exist in concepts table.
    
    Args:
        df_relationships: CONCEPT_RELATIONSHIP LazyFrame
        df_concepts: CONCEPT LazyFrame
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Get unique concept IDs
    concept_ids = set(df_concepts.select("concept_id").collect()["concept_id"].to_list())
    
    # Check concept_id_1
    rel_concept_1 = set(
        df_relationships.select("concept_id_1").collect()["concept_id_1"].to_list()
    )
    missing_1 = rel_concept_1 - concept_ids
    if missing_1:
        errors.append(f"Found {len(missing_1)} concept_id_1 values not in concepts table")
    
    # Check concept_id_2
    rel_concept_2 = set(
        df_relationships.select("concept_id_2").collect()["concept_id_2"].to_list()
    )
    missing_2 = rel_concept_2 - concept_ids
    if missing_2:
        errors.append(f"Found {len(missing_2)} concept_id_2 values not in concepts table")
    
    return len(errors) == 0, errors


def validate_ancestor_integrity(
    df_ancestors: pl.LazyFrame,
    df_concepts: pl.LazyFrame,
) -> Tuple[bool, List[str]]:
    """
    Validate that all ancestor/descendant IDs exist in concepts table.
    
    Args:
        df_ancestors: CONCEPT_ANCESTOR LazyFrame
        df_concepts: CONCEPT LazyFrame
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    concept_ids = set(df_concepts.select("concept_id").collect()["concept_id"].to_list())
    
    # Check ancestor IDs
    ancestor_ids = set(
        df_ancestors.select("ancestor_concept_id").collect()["ancestor_concept_id"].to_list()
    )
    missing_anc = ancestor_ids - concept_ids
    if missing_anc:
        errors.append(f"Found {len(missing_anc)} ancestor_concept_id values not in concepts table")
    
    # Check descendant IDs
    descendant_ids = set(
        df_ancestors.select("descendant_concept_id").collect()["descendant_concept_id"].to_list()
    )
    missing_desc = descendant_ids - concept_ids
    if missing_desc:
        errors.append(f"Found {len(missing_desc)} descendant_concept_id values not in concepts table")
    
    return len(errors) == 0, errors


def validate_vocabulary_mapping(
    df_concepts: pl.LazyFrame,
    df_vocabularies: pl.LazyFrame,
) -> Tuple[bool, List[str]]:
    """
    Validate that all vocabulary_id values in concepts exist in vocabularies table.
    
    Args:
        df_concepts: CONCEPT LazyFrame
        df_vocabularies: VOCABULARY LazyFrame
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    vocab_ids = set(
        df_vocabularies.select("vocabulary_id").collect()["vocabulary_id"].to_list()
    )
    concept_vocab_ids = set(
        df_concepts.select("vocabulary_id").collect()["vocabulary_id"].to_list()
    )
    
    missing_vocab = concept_vocab_ids - vocab_ids
    if missing_vocab:
        errors.append(f"Found {len(missing_vocab)} vocabulary_id values not in vocabularies table")
    
    return len(errors) == 0, errors


def validate_domain_mapping(
    df_concepts: pl.LazyFrame,
    df_domains: pl.LazyFrame,
) -> Tuple[bool, List[str]]:
    """
    Validate that all domain_id values in concepts exist in domains table.
    
    Args:
        df_concepts: CONCEPT LazyFrame
        df_domains: DOMAIN LazyFrame
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    domain_ids = set(df_domains.select("domain_id").collect()["domain_id"].to_list())
    concept_domain_ids = set(
        df_concepts.select("domain_id").collect()["domain_id"].to_list()
    )
    
    missing_domains = concept_domain_ids - domain_ids
    if missing_domains:
        errors.append(f"Found {len(missing_domains)} domain_id values not in domains table")
    
    return len(errors) == 0, errors


def get_data_quality_report(
    df_concepts: pl.LazyFrame,
    df_relationships: pl.LazyFrame,
    df_ancestors: pl.LazyFrame,
    df_vocabularies: pl.LazyFrame,
    df_domains: pl.LazyFrame,
) -> Dict[str, any]:
    """
    Generate a comprehensive data quality report.
    
    Args:
        df_concepts: CONCEPT LazyFrame
        df_relationships: CONCEPT_RELATIONSHIP LazyFrame
        df_ancestors: CONCEPT_ANCESTOR LazyFrame
        df_vocabularies: VOCABULARY LazyFrame
        df_domains: DOMAIN LazyFrame
        
    Returns:
        Dictionary with quality metrics
    """
    report = {}
    
    # Count records
    report["concepts_count"] = df_concepts.select(pl.len()).collect().item()
    report["relationships_count"] = df_relationships.select(pl.len()).collect().item()
    report["ancestors_count"] = df_ancestors.select(pl.len()).collect().item()
    report["vocabularies_count"] = df_vocabularies.select(pl.len()).collect().item()
    report["domains_count"] = df_domains.select(pl.len()).collect().item()
    
    # Check nulls in key fields
    concept_nulls = df_concepts.select([
        pl.col("concept_name").is_null().sum().alias("null_names"),
        pl.col("vocabulary_id").is_null().sum().alias("null_vocab"),
        pl.col("domain_id").is_null().sum().alias("null_domain"),
    ]).collect()
    report["concept_nulls"] = concept_nulls.to_dicts()[0]
    
    # Validate referential integrity
    rel_valid, rel_errors = validate_referential_integrity(df_relationships, df_concepts)
    report["relationships_valid"] = rel_valid
    report["relationship_errors"] = rel_errors
    
    anc_valid, anc_errors = validate_ancestor_integrity(df_ancestors, df_concepts)
    report["ancestors_valid"] = anc_valid
    report["ancestor_errors"] = anc_errors
    
    vocab_valid, vocab_errors = validate_vocabulary_mapping(df_concepts, df_vocabularies)
    report["vocabularies_valid"] = vocab_valid
    report["vocabulary_errors"] = vocab_errors
    
    domain_valid, domain_errors = validate_domain_mapping(df_concepts, df_domains)
    report["domains_valid"] = domain_valid
    report["domain_errors"] = domain_errors
    
    return report

