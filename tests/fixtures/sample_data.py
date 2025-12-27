"""Sample data fixtures for testing."""

import polars as pl


def create_sample_concepts() -> pl.DataFrame:
    """Create sample concept data."""
    return pl.DataFrame({
        "concept_id": [1, 2, 3, 4, 5],
        "concept_name": [
            "Myocardial infarction",
            "Heart attack",
            "Chest pain",
            "Hypertension",
            "Blood pressure",
        ],
        "vocabulary_id": ["SNOMED", "SNOMED", "SNOMED", "ICD10CM", "LOINC"],
        "domain_id": ["Condition", "Condition", "Condition", "Condition", "Measurement"],
        "concept_code": ["C001", "C002", "C003", "C004", "C005"],
        "concept_class_id": ["Clinical Finding", "Clinical Finding", "Clinical Finding", "Clinical Finding", "Lab Test"],
        "standard_concept": ["S", "S", "S", "S", "S"],
        "valid_start_date": ["2020-01-01"] * 5,
        "valid_end_date": ["2099-12-31"] * 5,
        "invalid_reason": [None] * 5,
    })


def create_sample_relationships() -> pl.DataFrame:
    """Create sample relationship data."""
    return pl.DataFrame({
        "concept_id_1": [1, 1, 2, 3, 4],
        "concept_id_2": [2, 3, 3, 4, 5],
        "relationship_id": [
            "concept_same_as",
            "is_a",
            "is_a",
            "associated_with",
            "has_finding_site",
        ],
        "valid_start_date": ["2020-01-01"] * 5,
        "valid_end_date": ["2099-12-31"] * 5,
        "invalid_reason": [None] * 5,
    })


def create_sample_ancestors() -> pl.DataFrame:
    """Create sample ancestor data."""
    return pl.DataFrame({
        "ancestor_concept_id": [10, 10, 11],
        "descendant_concept_id": [1, 2, 3],
        "min_levels_of_separation": [1, 1, 2],
        "max_levels_of_separation": [1, 1, 2],
    })


def create_sample_vocabularies() -> pl.DataFrame:
    """Create sample vocabulary data."""
    return pl.DataFrame({
        "vocabulary_id": ["SNOMED", "ICD10CM", "LOINC"],
        "vocabulary_name": [
            "SNOMED CT",
            "ICD10CM",
            "LOINC",
        ],
        "vocabulary_reference": ["http://snomed.org", "http://icd10cm.org", "http://loinc.org"],
        "vocabulary_version": ["2025-01", "2025", "2.80"],
        "vocabulary_concept_id": [100, 101, 102],
    })


def create_sample_domains() -> pl.DataFrame:
    """Create sample domain data."""
    return pl.DataFrame({
        "domain_id": ["Condition", "Measurement"],
        "domain_name": ["Condition", "Measurement"],
        "domain_concept_id": [200, 201],
    })

