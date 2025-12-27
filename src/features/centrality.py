"""NetworkX-based eigenvector centrality calculation."""

import networkx as nx
import polars as pl
from typing import Dict, List, Tuple


def build_graph_from_relationships(
    df_relationships: pl.DataFrame,
    df_concepts: pl.DataFrame,
) -> nx.DiGraph:
    """
    Build a directed graph from concept relationships.
    
    Args:
        df_relationships: DataFrame with concept_id_1, concept_id_2, weight
        df_concepts: DataFrame with concept_id
        
    Returns:
        NetworkX directed graph
    """
    G = nx.DiGraph()
    
    # Add all concept nodes
    concept_ids = df_concepts["concept_id"].to_list()
    G.add_nodes_from(concept_ids)
    
    # Add edges with weights
    for row in df_relationships.iter_rows(named=True):
        concept_id_1 = row["concept_id_1"]
        concept_id_2 = row["concept_id_2"]
        weight = row.get("weight", 1.0)
        
        if concept_id_1 in concept_ids and concept_id_2 in concept_ids:
            G.add_edge(concept_id_1, concept_id_2, weight=weight)
    
    return G


def calculate_eigenvector_centrality(
    G: nx.DiGraph,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> Dict[int, float]:
    """
    Calculate eigenvector centrality for all nodes.
    
    Args:
        G: NetworkX graph
        max_iter: Maximum iterations
        tol: Tolerance for convergence
        
    Returns:
        Dictionary mapping concept_id to centrality score
    """
    try:
        centrality = nx.eigenvector_centrality(G, max_iter=max_iter, tol=tol, weight="weight")
        return centrality
    except nx.PowerIterationFailedConvergence:
        # Fallback to degree centrality if eigenvector fails
        return dict(nx.degree_centrality(G))


def calculate_hierarchy_depth(
    df_ancestors: pl.DataFrame,
    concept_ids: List[int],
) -> Dict[int, int]:
    """
    Calculate hierarchy depth for concepts based on ancestor relationships.
    
    Args:
        df_ancestors: DataFrame with ancestor relationships
        concept_ids: List of concept IDs to calculate depth for
        
    Returns:
        Dictionary mapping concept_id to max depth
    """
    depth_map = {}
    
    for concept_id in concept_ids:
        # Find all ancestors and their levels
        ancestors = df_ancestors.filter(
            pl.col("descendant_concept_id") == concept_id
        )
        
        if len(ancestors) > 0:
            max_depth = ancestors["max_levels_of_separation"].max()
            depth_map[concept_id] = int(max_depth) if max_depth is not None else 0
        else:
            depth_map[concept_id] = 0
    
    return depth_map


def calculate_synonym_count(
    df_synonyms: pl.DataFrame,
    concept_ids: List[int],
) -> Dict[int, int]:
    """
    Calculate synonym count for concepts.
    
    Args:
        df_synonyms: DataFrame with concept_id and synonym_name
        concept_ids: List of concept IDs
        
    Returns:
        Dictionary mapping concept_id to synonym count
    """
    synonym_counts = (
        df_synonyms
        .group_by("concept_id")
        .agg(pl.len().alias("count"))
    )
    
    count_map = {}
    for row in synonym_counts.iter_rows(named=True):
        count_map[row["concept_id"]] = row["count"]
    
    # Fill in zeros for concepts without synonyms
    for concept_id in concept_ids:
        if concept_id not in count_map:
            count_map[concept_id] = 0
    
    return count_map

