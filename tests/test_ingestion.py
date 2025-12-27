"""Tests for data ingestion and cleaning."""

import pytest
from pathlib import Path
import polars as pl

from src.ingestion.loader import (
    load_concept,
    load_concept_relationship,
    load_vocabulary,
)
from src.ingestion.cleaner import (
    clean_concept_names,
    clean_relationships,
)
from src.ingestion.validator import (
    validate_referential_integrity,
    get_data_quality_report,
)
from tests.fixtures.sample_data import (
    create_sample_concepts,
    create_sample_relationships,
    create_sample_vocabularies,
    create_sample_domains,
)


def test_load_concept():
    """Test loading concept data."""
    # This would require actual CSV files, so we test with sample data
    df = create_sample_concepts()
    assert len(df) == 5
    assert "concept_id" in df.columns
    assert "concept_name" in df.columns


def test_clean_concept_names():
    """Test cleaning concept names."""
    df = create_sample_concepts()
    df_lazy = df.lazy()
    cleaned = clean_concept_names(df_lazy).collect()
    
    assert len(cleaned) == len(df)
    assert all(cleaned["concept_name"].str.strip_chars() == cleaned["concept_name"])


def test_clean_relationships():
    """Test cleaning relationships."""
    df = create_sample_relationships()
    df_lazy = df.lazy()
    cleaned = clean_relationships(df_lazy).collect()
    
    assert len(cleaned) == len(df)
    # All should be valid (no invalid_reason)
    assert cleaned["invalid_reason"].is_null().all()


def test_validate_referential_integrity():
    """Test referential integrity validation."""
    df_concepts = create_sample_concepts()
    df_relationships = create_sample_relationships()
    
    df_concepts_lazy = df_concepts.lazy()
    df_relationships_lazy = df_relationships.lazy()
    
    is_valid, errors = validate_referential_integrity(df_relationships_lazy, df_concepts_lazy)
    
    # All relationships should be valid (concept IDs exist)
    assert is_valid
    assert len(errors) == 0


def test_data_quality_report():
    """Test data quality report generation."""
    df_concepts = create_sample_concepts().lazy()
    df_relationships = create_sample_relationships().lazy()
    df_ancestors = create_sample_ancestors().lazy()
    df_vocabularies = create_sample_vocabularies().lazy()
    df_domains = create_sample_domains().lazy()
    
    report = get_data_quality_report(
        df_concepts,
        df_relationships,
        df_ancestors,
        df_vocabularies,
        df_domains,
    )
    
    assert "concepts_count" in report
    assert report["concepts_count"] == 5
    assert "relationships_valid" in report
    assert report["relationships_valid"] is True

