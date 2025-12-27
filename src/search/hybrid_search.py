"""Hybrid search query builder combining graph traversal and vector similarity."""

from typing import Dict, List, Set, Optional, Any
import numpy as np

from src.search.query_executor import HybridSearchExecutor
from src.search.vector_search import calculate_centroid
from src.search.graph_traversal import calculate_path_density


class HybridSearchBuilder:
    """Builder for hybrid search queries."""
    
    def __init__(self, executor: Optional[HybridSearchExecutor] = None):
        """
        Initialize hybrid search builder.
        
        Args:
            executor: HybridSearchExecutor instance
        """
        self.executor = executor or HybridSearchExecutor()
        self.query_params = {}
    
    def with_query_embedding(self, embedding: np.ndarray) -> "HybridSearchBuilder":
        """Set query embedding for semantic search."""
        self.query_params["query_embedding"] = embedding
        return self
    
    def with_domain_cluster(self, cluster_id: str) -> "HybridSearchBuilder":
        """Set domain cluster ID."""
        self.query_params["domain_cluster_id"] = cluster_id
        return self
    
    def with_domain_concepts(self, concept_ids: Set[int]) -> "HybridSearchBuilder":
        """Set domain cluster as set of concept IDs."""
        self.query_params["domain_concepts"] = concept_ids
        return self
    
    def with_limit(self, limit: int) -> "HybridSearchBuilder":
        """Set result limit."""
        self.query_params["limit"] = limit
        return self
    
    def with_min_score(self, min_score: float) -> "HybridSearchBuilder":
        """Set minimum score threshold."""
        self.query_params["min_score"] = min_score
        return self
    
    def with_weights(
        self,
        vector_weight: float = 0.5,
        graph_weight: float = 0.5,
    ) -> "HybridSearchBuilder":
        """Set weights for vector and graph components."""
        self.query_params["vector_weight"] = vector_weight
        self.query_params["graph_weight"] = graph_weight
        return self
    
    def execute(self) -> List[Dict[str, Any]]:
        """
        Execute the hybrid search query.
        
        Returns:
            List of search results
        """
        if "domain_cluster_id" in self.query_params:
            # Search by domain cluster ID
            return self.executor.search_by_domain_cluster(
                domain_cluster_id=self.query_params["domain_cluster_id"],
                query_embedding=self.query_params.get("query_embedding"),
                limit=self.query_params.get("limit", 10),
                min_score=self.query_params.get("min_score", 0.0),
            )
        elif "domain_concepts" in self.query_params and "query_embedding" in self.query_params:
            # Hybrid search with domain concepts and query embedding
            # Note: This requires relationship_map which should be loaded separately
            # For now, use vector search only
            return self.executor.vector_search(
                query_embedding=self.query_params["query_embedding"],
                limit=self.query_params.get("limit", 10),
                threshold=self.query_params.get("min_score", 0.0),
            )
        elif "query_embedding" in self.query_params:
            # Vector search only
            return self.executor.vector_search(
                query_embedding=self.query_params["query_embedding"],
                limit=self.query_params.get("limit", 10),
                threshold=self.query_params.get("min_score", 0.0),
            )
        else:
            raise ValueError("Invalid query parameters: need domain_cluster_id or query_embedding")


def search_concepts_by_domain(
    domain_cluster_id: str,
    limit: int = 10,
    min_score: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    Search for concepts by domain cluster ID.
    
    Args:
        domain_cluster_id: Domain cluster identifier
        limit: Maximum number of results
        min_score: Minimum relevance score
        
    Returns:
        List of concept results
    """
    builder = HybridSearchBuilder()
    return (
        builder
        .with_domain_cluster(domain_cluster_id)
        .with_limit(limit)
        .with_min_score(min_score)
        .execute()
    )


def search_concepts_by_semantic_similarity(
    query_embedding: np.ndarray,
    limit: int = 10,
    threshold: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    Search for concepts by semantic similarity.
    
    Args:
        query_embedding: Query embedding vector
        limit: Maximum number of results
        threshold: Minimum similarity threshold
        
    Returns:
        List of concept results
    """
    builder = HybridSearchBuilder()
    return (
        builder
        .with_query_embedding(query_embedding)
        .with_limit(limit)
        .with_min_score(threshold)
        .execute()
    )

