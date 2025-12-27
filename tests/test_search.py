"""Tests for hybrid search functionality."""

import pytest
import numpy as np

from src.search.graph_traversal import find_paths_3hop, calculate_path_density
from src.search.vector_search import cosine_similarity, calculate_centroid
from src.search.hybrid_search import HybridSearchBuilder


def test_find_paths_3hop():
    """Test 3-hop path finding."""
    start_id = 1
    target_ids = {3, 4}
    relationship_map = {
        1: [(2, "is_a", 0.8)],
        2: [(3, "is_a", 0.8), (4, "associated_with", 0.5)],
    }
    
    paths = find_paths_3hop(start_id, target_ids, relationship_map, max_hops=3)
    
    assert len(paths) > 0
    assert all(hop_count <= 3 for hop_count in paths.values())


def test_cosine_similarity():
    """Test cosine similarity calculation."""
    vec1 = np.array([1.0, 0.0, 0.0])
    vec2 = np.array([1.0, 0.0, 0.0])
    
    similarity = cosine_similarity(vec1, vec2)
    assert abs(similarity - 1.0) < 0.01  # Should be 1.0 for identical vectors
    
    vec3 = np.array([0.0, 1.0, 0.0])
    similarity2 = cosine_similarity(vec1, vec3)
    assert abs(similarity2) < 0.01  # Should be 0.0 for orthogonal vectors


def test_calculate_centroid():
    """Test centroid calculation."""
    embeddings = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ]
    
    centroid = calculate_centroid(embeddings)
    
    assert len(centroid) == 3
    assert np.allclose(centroid, np.array([1/3, 1/3, 1/3]), atol=0.1)


def test_path_density():
    """Test path density calculation."""
    concept_id = 1
    domain_cluster = {2, 3, 4}
    relationship_map = {
        1: [(2, "is_a", 0.8)],
        2: [(3, "is_a", 0.8)],
    }
    
    density = calculate_path_density(concept_id, domain_cluster, relationship_map, max_hops=3)
    
    assert 0.0 <= density <= 1.0

