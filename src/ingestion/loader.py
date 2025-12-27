"""Polars-based CSV readers with chunking support for large UMLS files."""

import polars as pl
from pathlib import Path
from typing import Optional


def load_csv_chunked(
    file_path: Path,
    chunk_size: int = 100_000,
    separator: str = "\t",
    infer_schema_length: Optional[int] = 10_000,
) -> pl.LazyFrame:
    """
    Load a large CSV file as a LazyFrame with chunking support.
    
    Args:
        file_path: Path to the CSV file
        chunk_size: Number of rows to process per chunk
        separator: CSV separator (default: tab for UMLS files)
        infer_schema_length: Number of rows to use for schema inference
        
    Returns:
        LazyFrame for lazy evaluation
    """
    return pl.scan_csv(
        file_path,
        separator=separator,
        infer_schema_length=infer_schema_length,
        try_parse_dates=True,
        null_values=["", "NULL", "null", "None"],
    )


def load_concept(file_path: Path) -> pl.LazyFrame:
    """Load CONCEPT.csv file."""
    return load_csv_chunked(file_path).select([
        pl.col("concept_id").cast(pl.Int64),
        pl.col("concept_name").str.strip_chars(),
        pl.col("domain_id").str.strip_chars(),
        pl.col("vocabulary_id").str.strip_chars(),
        pl.col("concept_class_id").str.strip_chars(),
        pl.col("standard_concept").str.strip_chars(),
        pl.col("concept_code").str.strip_chars(),
        pl.col("valid_start_date"),
        pl.col("valid_end_date"),
        pl.col("invalid_reason").str.strip_chars(),
    ])


def load_concept_relationship(file_path: Path) -> pl.LazyFrame:
    """Load CONCEPT_RELATIONSHIP.csv file."""
    return load_csv_chunked(file_path).select([
        pl.col("concept_id_1").cast(pl.Int64),
        pl.col("concept_id_2").cast(pl.Int64),
        pl.col("relationship_id").str.strip_chars(),
        pl.col("valid_start_date"),
        pl.col("valid_end_date"),
        pl.col("invalid_reason").str.strip_chars(),
    ])


def load_concept_ancestor(file_path: Path) -> pl.LazyFrame:
    """Load CONCEPT_ANCESTOR.csv file."""
    return load_csv_chunked(file_path).select([
        pl.col("ancestor_concept_id").cast(pl.Int64),
        pl.col("descendant_concept_id").cast(pl.Int64),
        pl.col("min_levels_of_separation").cast(pl.Int32),
        pl.col("max_levels_of_separation").cast(pl.Int32),
    ])


def load_concept_synonym(file_path: Path) -> pl.LazyFrame:
    """Load CONCEPT_SYNONYM.csv file."""
    return load_csv_chunked(file_path).select([
        pl.col("concept_id").cast(pl.Int64),
        pl.col("concept_synonym_name").str.strip_chars(),
        pl.col("language_concept_id").cast(pl.Int64),
    ])


def load_vocabulary(file_path: Path) -> pl.LazyFrame:
    """Load VOCABULARY.csv file."""
    return load_csv_chunked(file_path).select([
        pl.col("vocabulary_id").str.strip_chars(),
        pl.col("vocabulary_name").str.strip_chars(),
        pl.col("vocabulary_reference").str.strip_chars(),
        pl.col("vocabulary_version").str.strip_chars(),
        pl.col("vocabulary_concept_id").cast(pl.Int64),
    ])


def load_domain(file_path: Path) -> pl.LazyFrame:
    """Load DOMAIN.csv file."""
    return load_csv_chunked(file_path).select([
        pl.col("domain_id").str.strip_chars(),
        pl.col("domain_name").str.strip_chars(),
        pl.col("domain_concept_id").cast(pl.Int64),
    ])


def load_relationship(file_path: Path) -> pl.LazyFrame:
    """Load RELATIONSHIP.csv file."""
    return load_csv_chunked(file_path).select([
        pl.col("relationship_id").str.strip_chars(),
        pl.col("relationship_name").str.strip_chars(),
        pl.col("is_hierarchical").cast(pl.Int8),
        pl.col("defines_ancestry").cast(pl.Int8),
        pl.col("reverse_relationship_id").str.strip_chars(),
        pl.col("relationship_concept_id").cast(pl.Int64),
    ])

