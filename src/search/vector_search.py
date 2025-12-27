"""Vector similarity search using SurrealDB MTREE index."""

import numpy as np
from typing import List, Dict, Tuple, Optional


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity score [-1.0, 1.0]
    """
    if len(vec1) == 0 or len(vec2) == 0:
        return 0.0
    
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def calculate_centroid(embeddings: List[np.ndarray]) -> np.ndarray:
    """
    Calculate centroid of a list of embeddings.
    
    Args:
        embeddings: List of embedding vectors
        
    Returns:
        Centroid vector
    """
    if len(embeddings) == 0:
        return np.array([])
    
    centroid = np.mean(embeddings, axis=0)
    
    # Normalize
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm
    
    return centroid


def vector_search_query(
    query_embedding: np.ndarray,
    limit: int = 10,
    threshold: float = 0.0,
) -> str:
    """
    Generate SurrealDB vector search query using MTREE index.
    
    Args:
        query_embedding: Query embedding vector
        limit: Maximum number of results
        threshold: Minimum similarity threshold
        
    Returns:
        SurrealQL query string
    """
    # Convert embedding to list for SurrealDB
    embedding_list = query_embedding.tolist()
    
    # SurrealDB vector search query
    query = f"""
    SELECT 
        concept_id,
        source_authority,
        centrality,
        vector::similarity::cosine(semantic_embed, {embedding_list}) AS similarity
    FROM metric
    WHERE vector::similarity::cosine(semantic_embed, {embedding_list}) >= {threshold}
    ORDER BY similarity DESC
    LIMIT {limit}
    """
    
    return query


def find_similar_concepts(
    query_embedding: np.ndarray,
    concept_embeddings: Dict[int, np.ndarray],
    limit: int = 10,
    threshold: float = 0.0,
) -> List[Tuple[int, float]]:
    """
    Find similar concepts using cosine similarity (in-memory version).
    
    Args:
        query_embedding: Query embedding vector
        concept_embeddings: Map from concept_id to embedding
        limit: Maximum number of results
        threshold: Minimum similarity threshold
        
    Returns:
        List of (concept_id, similarity_score) tuples, sorted by similarity
    """
    similarities = []
    
    for concept_id, embedding in concept_embeddings.items():
        similarity = cosine_similarity(query_embedding, embedding)
        
        if similarity >= threshold:
            similarities.append((concept_id, similarity))
    
    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities[:limit]

