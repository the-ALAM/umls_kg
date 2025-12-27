"""Tests for relevance scoring."""

import pytest
import numpy as np

from src.scoring.formula import (
    calculate_s_struct,
    calculate_s_sem,
    calculate_s_density,
    sigmoid,
)
from src.scoring.relevance import RelevanceScorer, build_relationship_map, build_ancestor_map
from tests.fixtures.sample_data import create_sample_relationships, create_sample_ancestors


def test_s_struct():
    """Test structural score calculation."""
    concept_id = 1
    domain_cluster = {2, 3}
    relationship_map = {
        1: [(2, "concept_same_as", 1.0), (3, "is_a", 0.8)],
    }
    ancestor_map = {}
    
    score = calculate_s_struct(concept_id, domain_cluster, relationship_map, ancestor_map)
    assert score == 1.0  # Direct relationship


def test_s_sem():
    """Test semantic similarity calculation."""
    # Create normalized embeddings
    embedding1 = np.array([1.0, 0.0, 0.0])
    embedding2 = np.array([1.0, 0.0, 0.0])
    
    score = calculate_s_sem(embedding1, embedding2)
    assert 0.0 <= score <= 1.0


def test_s_density():
    """Test path density calculation."""
    concept_id = 1
    domain_cluster = {2, 3}
    graph_paths = {
        1: {2: 1, 3: 2},  # 1 hop to 2, 2 hops to 3
    }
    
    score = calculate_s_density(concept_id, domain_cluster, graph_paths)
    assert 0.0 <= score <= 1.0


def test_sigmoid():
    """Test sigmoid function."""
    assert sigmoid(0) == 0.5
    assert sigmoid(-10) < 0.1
    assert sigmoid(10) > 0.9
    assert 0.0 <= sigmoid(5) <= 1.0


def test_relevance_scorer():
    """Test relevance scorer."""
    scorer = RelevanceScorer(alpha=0.4, beta=0.3, gamma=0.2, delta=0.1)
    
    concept_id = 1
    domain_cluster = {2, 3}
    relationship_map = {1: [(2, "is_a", 0.8)]}
    ancestor_map = {}
    concept_embedding = np.array([1.0, 0.0, 0.0])
    domain_centroid = np.array([1.0, 0.0, 0.0])
    graph_paths = {1: {2: 1}}
    source_authority = 1.0
    
    result = scorer.calculate_relevance(
        concept_id,
        domain_cluster,
        relationship_map,
        ancestor_map,
        concept_embedding,
        domain_centroid,
        graph_paths,
        source_authority,
    )
    
    assert "score" in result
    assert 0.0 <= result["score"] <= 1.0
    assert "s_struct" in result
    assert "s_sem" in result
    assert "s_density" in result
    assert "s_authority" in result

