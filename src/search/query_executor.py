"""Execute combined SurrealDB queries for hybrid search."""

from typing import List, Dict, Any, Set, Optional
import numpy as np

from src.database.surreal_client import SurrealDBSync
from src.search.vector_search import vector_search_query, cosine_similarity
from src.search.graph_traversal import find_paths_3hop, calculate_path_density


class HybridSearchExecutor:
    """Execute hybrid search queries combining graph traversal and vector similarity."""
    
    def __init__(self, client: Optional[SurrealDBSync] = None):
        """
        Initialize hybrid search executor.
        
        Args:
            client: SurrealDB client (creates new if None)
        """
        self.client = client or SurrealDBSync()
        if not hasattr(self.client, '_connected') or not getattr(self.client, '_connected', False):
            self.client.connect()
    
    def search_by_domain_cluster(
        self,
        domain_cluster_id: str,
        query_embedding: Optional[np.ndarray] = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Search for concepts relevant to a domain cluster.
        
        Args:
            domain_cluster_id: Domain cluster identifier
            query_embedding: Optional query embedding for semantic search
            limit: Maximum number of results
            min_score: Minimum relevance score
            
        Returns:
            List of concept results with scores
        """
        # Query relevance scores for this domain cluster
        query = f"""
        SELECT 
            concept_id,
            score,
            s_struct,
            s_sem,
            s_density,
            s_authority
        FROM relevance_score
        WHERE domain_cluster_id = '{domain_cluster_id}'
        AND score >= {min_score}
        ORDER BY score DESC
        LIMIT {limit}
        """
        
        results = self.client.query(query)
        
        # Format results
        formatted_results = []
        for result in results:
            if isinstance(result, list) and len(result) > 0:
                for item in result:
                    formatted_results.append({
                        "concept_id": item.get("concept_id"),
                        "score": item.get("score"),
                        "s_struct": item.get("s_struct"),
                        "s_sem": item.get("s_sem"),
                        "s_density": item.get("s_density"),
                        "s_authority": item.get("s_authority"),
                    })
            elif isinstance(result, dict):
                formatted_results.append(result)
        
        return formatted_results[:limit]
    
    def vector_search(
        self,
        query_embedding: np.ndarray,
        limit: int = 10,
        threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search.
        
        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results
            threshold: Minimum similarity threshold
            
        Returns:
            List of concept results with similarity scores
        """
        query = vector_search_query(query_embedding, limit, threshold)
        results = self.client.query(query)
        
        formatted_results = []
        for result in results:
            if isinstance(result, list) and len(result) > 0:
                formatted_results.extend(result)
            elif isinstance(result, dict):
                formatted_results.append(result)
        
        return formatted_results[:limit]
    
    def graph_search(
        self,
        start_concept_id: int,
        target_concept_ids: Set[int],
        relationship_map: Dict[int, List[tuple]],
        max_hops: int = 3,
    ) -> Dict[int, int]:
        """
        Perform graph traversal search.
        
        Args:
            start_concept_id: Starting concept ID
            target_concept_ids: Set of target concept IDs
            relationship_map: Map from concept_id to relationships
            max_hops: Maximum number of hops
            
        Returns:
            Dictionary mapping target_concept_id to hop_count
        """
        return find_paths_3hop(start_concept_id, target_concept_ids, relationship_map, max_hops)
    
    def hybrid_search(
        self,
        query_embedding: np.ndarray,
        domain_cluster: Set[int],
        relationship_map: Dict[int, List[tuple]],
        limit: int = 10,
        vector_weight: float = 0.5,
        graph_weight: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining vector and graph methods.
        
        Args:
            query_embedding: Query embedding vector
            domain_cluster: Set of concept IDs in domain cluster
            relationship_map: Map from concept_id to relationships
            limit: Maximum number of results
            vector_weight: Weight for vector similarity scores
            graph_weight: Weight for graph traversal scores
            
        Returns:
            List of concept results with combined scores
        """
        # Vector search
        vector_results = self.vector_search(query_embedding, limit=limit * 2)
        vector_scores = {
            result["concept_id"]: result.get("similarity", 0.0)
            for result in vector_results
        }
        
        # Graph search (for concepts in domain cluster)
        graph_scores = {}
        for concept_id in domain_cluster:
            if concept_id in relationship_map:
                density = calculate_path_density(concept_id, domain_cluster, relationship_map)
                graph_scores[concept_id] = density
        
        # Combine scores
        all_concept_ids = set(vector_scores.keys()) | set(graph_scores.keys())
        combined_results = []
        
        for concept_id in all_concept_ids:
            vector_score = vector_scores.get(concept_id, 0.0)
            graph_score = graph_scores.get(concept_id, 0.0)
            
            # Normalize scores to [0, 1]
            combined_score = (vector_weight * vector_score) + (graph_weight * graph_score)
            
            combined_results.append({
                "concept_id": concept_id,
                "combined_score": combined_score,
                "vector_score": vector_score,
                "graph_score": graph_score,
            })
        
        # Sort by combined score
        combined_results.sort(key=lambda x: x["combined_score"], reverse=True)
        
        return combined_results[:limit]

