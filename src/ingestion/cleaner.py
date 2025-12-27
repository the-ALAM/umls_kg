"""Data normalization and cleaning utilities for UMLS data."""

import polars as pl
from typing import Optional


def normalize_string_column(df: pl.LazyFrame, column: str) -> pl.LazyFrame:
    """
    Normalize a string column by trimming whitespace and handling nulls.
    
    Args:
        df: Input LazyFrame
        column: Column name to normalize
        
    Returns:
        LazyFrame with normalized column
    """
    return df.with_columns([
        pl.col(column)
        .str.strip_chars()
        .str.to_lowercase()
        .fill_null("")
    ])


def clean_concept_names(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Clean and normalize concept names.
    
    Args:
        df: CONCEPT LazyFrame
        
    Returns:
        Cleaned LazyFrame
    """
    return df.with_columns([
        pl.col("concept_name")
        .str.strip_chars()
        .str.replace_all(r"\s+", " ")  # Normalize whitespace
        .fill_null(""),
        pl.col("concept_code").str.strip_chars().fill_null(""),
        pl.col("vocabulary_id").str.strip_chars().fill_null(""),
        pl.col("domain_id").str.strip_chars().fill_null(""),
    ])


def clean_relationships(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Clean and normalize relationship data.
    
    Args:
        df: CONCEPT_RELATIONSHIP LazyFrame
        
    Returns:
        Cleaned LazyFrame
    """
    return df.with_columns([
        pl.col("relationship_id").str.strip_chars().fill_null(""),
        pl.col("invalid_reason").str.strip_chars().fill_null(""),
    ]).filter(
        # Filter out invalid relationships
        (pl.col("invalid_reason").is_null()) | (pl.col("invalid_reason") == "")
    )


def clean_ancestors(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Clean and normalize ancestor data.
    
    Args:
        df: CONCEPT_ANCESTOR LazyFrame
        
    Returns:
        Cleaned LazyFrame
    """
    return df.filter(
        # Ensure valid hierarchy levels
        (pl.col("min_levels_of_separation") >= 1) &
        (pl.col("max_levels_of_separation") >= pl.col("min_levels_of_separation"))
    )


def clean_synonyms(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Clean and normalize concept synonyms.
    
    Args:
        df: CONCEPT_SYNONYM LazyFrame
        
    Returns:
        Cleaned LazyFrame
    """
    return df.with_columns([
        pl.col("concept_synonym_name")
        .str.strip_chars()
        .str.replace_all(r"\s+", " ")
        .fill_null(""),
    ]).filter(
        # Remove empty synonyms
        pl.col("concept_synonym_name") != ""
    )


def combine_concept_text(df_concept: pl.LazyFrame, df_synonym: pl.LazyFrame) -> pl.LazyFrame:
    """
    Combine concept names with synonyms for embedding generation.
    
    Args:
        df_concept: CONCEPT LazyFrame
        df_synonym: CONCEPT_SYNONYM LazyFrame
        
    Returns:
        LazyFrame with combined text field
    """
    # Aggregate synonyms per concept
    synonyms_agg = (
        df_synonym
        .group_by("concept_id")
        .agg([
            pl.col("concept_synonym_name").unique().sort().list.join("; ")
        ])
        .rename({"concept_synonym_name": "synonyms"})
    )
    
    # Join with concepts
    return (
        df_concept
        .join(synonyms_agg, on="concept_id", how="left")
        .with_columns([
            pl.when(pl.col("synonyms").is_null())
            .then(pl.lit(""))
            .otherwise(pl.col("synonyms"))
            .alias("synonyms")
        ])
        .with_columns([
            (pl.col("concept_name") + " " + pl.col("synonyms"))
            .str.strip_chars()
            .alias("concept_text")
        ])
    )

