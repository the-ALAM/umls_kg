"""Relevance score calculation using sigmoid-weighted formula."""

import numpy as np
from typing import Dict, Set, List, Tuple
from datetime import datetime

from src.scoring.formula import (
    calculate_s_struct,
    calculate_s_sem,
    calculate_s_density,
    calculate_s_authority,
    sigmoid,
)


class RelevanceScorer:
    """Calculate relevance scores between concepts and domain clusters."""
    
    def __init__(
        self,
        alpha: float = 0.4,  # Weight for S_struct
        beta: float = 0.3,   # Weight for S_sem
        gamma: float = 0.2,  # Weight for S_density
        delta: float = 0.1,  # Weight for S_authority
    ):
        """
        Initialize relevance scorer with weights.
        
        Args:
            alpha: Weight for structural score
            beta: Weight for semantic score
            gamma: Weight for density score
            delta: Weight for authority score
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
    
    def calculate_relevance(
        self,
        concept_id: int,
        domain_cluster: Set[int],
        relationship_map: Dict[int, List[Tuple[int, str, float]]],
        ancestor_map: Dict[int, Set[int]],
        concept_embedding: np.ndarray,
        domain_cluster_centroid: np.ndarray,
        graph_paths: Dict[int, Dict[int, int]],
        source_authority: float,
    ) -> Dict[str, float]:
        """
        Calculate relevance score for a concept against a domain cluster.
        
        Args:
            concept_id: Concept ID to score
            domain_cluster: Set of concept IDs in domain cluster
            relationship_map: Map from concept_id to relationships
            ancestor_map: Map from concept_id to ancestor IDs
            concept_embedding: Concept embedding vector
            domain_cluster_centroid: Domain cluster centroid
            graph_paths: Graph path information
            source_authority: Authority score
            
        Returns:
            Dictionary with score components and final score
        """
        # Calculate component scores
        s_struct = calculate_s_struct(concept_id, domain_cluster, relationship_map, ancestor_map)
        s_sem = calculate_s_sem(concept_embedding, domain_cluster_centroid)
        s_density = calculate_s_density(concept_id, domain_cluster, graph_paths)
        s_authority = calculate_s_authority(source_authority)
        
        # Weighted sum
        weighted_sum = (
            self.alpha * s_struct +
            self.beta * s_sem +
            self.gamma * s_density +
            self.delta * s_authority
        )
        
        # Sigmoid squashing
        final_score = sigmoid(weighted_sum)
        
        return {
            "score": final_score,
            "s_struct": s_struct,
            "s_sem": s_sem,
            "s_density": s_density,
            "s_authority": s_authority,
        }
    
    def calculate_domain_cluster_centroid(
        self,
        domain_cluster: Set[int],
        embeddings: Dict[int, np.ndarray],
    ) -> np.ndarray:
        """
        Calculate centroid embedding for a domain cluster.
        
        Args:
            domain_cluster: Set of concept IDs in cluster
            embeddings: Map from concept_id to embedding
            
        Returns:
            Centroid embedding vector
        """
        cluster_embeddings = [
            embeddings[concept_id]
            for concept_id in domain_cluster
            if concept_id in embeddings and len(embeddings[concept_id]) > 0
        ]
        
        if len(cluster_embeddings) == 0:
            # Return zero vector if no embeddings
            if len(embeddings) > 0:
                # Use first embedding's dimension
                first_embedding = next(iter(embeddings.values()))
                return np.zeros_like(first_embedding)
            return np.array([])
        
        # Average embeddings
        centroid = np.mean(cluster_embeddings, axis=0)
        
        # Normalize
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        
        return centroid


def build_relationship_map(
    relationships: List[Dict],
) -> Dict[int, List[Tuple[int, str, float]]]:
    """
    Build relationship map from relationship data.
    
    Args:
        relationships: List of relationship dictionaries
        
    Returns:
        Map from concept_id to list of (target_id, relationship_type, weight)
    """
    rel_map = {}
    
    for rel in relationships:
        concept_id_1 = rel["concept_id_1"]
        concept_id_2 = rel["concept_id_2"]
        relationship_id = rel.get("relationship_id", "")
        weight = rel.get("weight", 1.0)
        
        if concept_id_1 not in rel_map:
            rel_map[concept_id_1] = []
        rel_map[concept_id_1].append((concept_id_2, relationship_id, weight))
    
    return rel_map


def build_ancestor_map(
    ancestors: List[Dict],
) -> Dict[int, Set[int]]:
    """
    Build ancestor map from ancestor data.
    
    Args:
        ancestors: List of ancestor dictionaries
        
    Returns:
        Map from concept_id to set of ancestor IDs
    """
    ancestor_map = {}
    
    for anc in ancestors:
        descendant_id = anc["descendant_concept_id"]
        ancestor_id = anc["ancestor_concept_id"]
        
        if descendant_id not in ancestor_map:
            ancestor_map[descendant_id] = set()
        ancestor_map[descendant_id].add(ancestor_id)
    
    return ancestor_map

