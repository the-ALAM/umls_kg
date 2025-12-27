"""Score component calculations for relevance scoring."""

import numpy as np
from typing import List, Dict, Set


def calculate_s_struct(
    concept_id: int,
    domain_cluster: Set[int],
    relationship_map: Dict[int, List[tuple]],
    ancestor_map: Dict[int, Set[int]],
) -> float:
    """
    Calculate structural score (S_struct).
    
    Binary score: 1.0 for direct mapping/is_a, 0.5 for ancestor, 0.0 otherwise.
    
    Args:
        concept_id: Concept ID to score
        domain_cluster: Set of concept IDs in the domain cluster
        relationship_map: Map from concept_id to list of (target_id, relationship_type, weight)
        ancestor_map: Map from concept_id to set of ancestor IDs
        
    Returns:
        Structural score [0.0, 1.0]
    """
    # Check if concept is directly in cluster
    if concept_id in domain_cluster:
        return 1.0
    
    # Check for direct relationships (mapping/is_a)
    if concept_id in relationship_map:
        for target_id, rel_type, weight in relationship_map[concept_id]:
            if target_id in domain_cluster:
                # Check if it's a mapping or is_a relationship
                if rel_type in ["mapped_from", "maps_to", "concept_same_as", "is_a", "isa"]:
                    return 1.0
    
    # Check if concept is an ancestor of any cluster member
    if concept_id in ancestor_map:
        if ancestor_map[concept_id] & domain_cluster:
            return 0.5
    
    # Check if any cluster member is an ancestor of this concept
    for cluster_id in domain_cluster:
        if cluster_id in ancestor_map:
            if concept_id in ancestor_map[cluster_id]:
                return 0.5
    
    return 0.0


def calculate_s_sem(
    concept_embedding: np.ndarray,
    domain_cluster_centroid: np.ndarray,
) -> float:
    """
    Calculate semantic similarity score (S_sem).
    
    Cosine similarity between concept embedding and domain cluster centroid.
    
    Args:
        concept_embedding: Concept embedding vector
        domain_cluster_centroid: Domain cluster centroid embedding
        
    Returns:
        Semantic similarity score [0.0, 1.0]
    """
    if len(concept_embedding) == 0 or len(domain_cluster_centroid) == 0:
        return 0.0
    
    # Cosine similarity (embeddings should already be normalized)
    dot_product = np.dot(concept_embedding, domain_cluster_centroid)
    
    # Clamp to [0, 1] range (cosine similarity is [-1, 1], but normalized embeddings should be [0, 1])
    return max(0.0, min(1.0, (dot_product + 1.0) / 2.0))


def calculate_s_density(
    concept_id: int,
    domain_cluster: Set[int],
    graph_paths: Dict[int, Dict[int, int]],
) -> float:
    """
    Calculate path density score (S_density).
    
    Proportion of nodes in domain cluster reachable within 3 hops, weighted by distance decay (1/d²).
    
    Args:
        concept_id: Concept ID to score
        domain_cluster: Set of concept IDs in the domain cluster
        graph_paths: Map from concept_id to map of (target_id -> hop_count)
        
    Returns:
        Density score [0.0, 1.0]
    """
    if concept_id not in graph_paths:
        return 0.0
    
    paths_from_concept = graph_paths[concept_id]
    total_score = 0.0
    
    for cluster_id in domain_cluster:
        if cluster_id in paths_from_concept:
            hop_count = paths_from_concept[cluster_id]
            if hop_count <= 3:
                # Distance decay: 1 / (hop_count + 1)²
                weight = 1.0 / ((hop_count + 1) ** 2)
                total_score += weight
    
    # Normalize by cluster size
    if len(domain_cluster) > 0:
        return total_score / len(domain_cluster)
    
    return 0.0


def calculate_s_authority(
    source_authority: float,
) -> float:
    """
    Calculate authority score (S_authority).
    
    Direct pass-through of source_authority from metric sidecar.
    
    Args:
        source_authority: Authority score from metric sidecar
        
    Returns:
        Authority score [0.0, 1.0]
    """
    return float(source_authority)


def sigmoid(x: float) -> float:
    """
    Sigmoid function for non-linear squashing.
    
    Args:
        x: Input value
        
    Returns:
        Sigmoid output [0.0, 1.0]
    """
    return 1.0 / (1.0 + np.exp(-x))

