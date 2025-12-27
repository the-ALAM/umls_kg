"""Graph path finding with 3-hop limit."""

from typing import Dict, List, Set, Tuple, Optional
from collections import deque


def find_paths_3hop(
    start_concept_id: int,
    target_concept_ids: Set[int],
    relationship_map: Dict[int, List[Tuple[int, str, float]]],
    max_hops: int = 3,
) -> Dict[int, int]:
    """
    Find shortest paths from start concept to target concepts within max_hops.
    
    Args:
        start_concept_id: Starting concept ID
        target_concept_ids: Set of target concept IDs
        relationship_map: Map from concept_id to list of (target_id, relationship_type, weight)
        max_hops: Maximum number of hops (default: 3)
        
    Returns:
        Dictionary mapping target_concept_id to hop_count
    """
    if start_concept_id not in relationship_map:
        return {}
    
    # BFS to find shortest paths
    queue = deque([(start_concept_id, 0)])  # (concept_id, hop_count)
    visited = {start_concept_id: 0}
    results = {}
    
    while queue:
        current_id, hop_count = queue.popleft()
        
        if hop_count >= max_hops:
            continue
        
        if current_id not in relationship_map:
            continue
        
        # Explore neighbors
        for target_id, rel_type, weight in relationship_map[current_id]:
            if target_id in visited and visited[target_id] <= hop_count + 1:
                continue
            
            visited[target_id] = hop_count + 1
            
            # Check if this is a target
            if target_id in target_concept_ids:
                results[target_id] = hop_count + 1
            
            # Continue traversal
            queue.append((target_id, hop_count + 1))
    
    return results


def find_paths_with_weights(
    start_concept_id: int,
    target_concept_ids: Set[int],
    relationship_map: Dict[int, List[Tuple[int, str, float]]],
    max_hops: int = 3,
) -> Dict[int, Tuple[int, float]]:
    """
    Find paths with weighted distances.
    
    Args:
        start_concept_id: Starting concept ID
        target_concept_ids: Set of target concept IDs
        relationship_map: Map from concept_id to relationships
        max_hops: Maximum number of hops
        
    Returns:
        Dictionary mapping target_concept_id to (hop_count, weighted_distance)
    """
    if start_concept_id not in relationship_map:
        return {}
    
    # Dijkstra-like algorithm with hop limit
    queue = [(0.0, start_concept_id, 0)]  # (weighted_distance, concept_id, hop_count)
    visited = {}  # concept_id -> (hop_count, weighted_distance)
    results = {}
    
    import heapq
    heapq.heapify(queue)
    
    while queue:
        weighted_dist, current_id, hop_count = heapq.heappop(queue)
        
        if hop_count >= max_hops:
            continue
        
        if current_id in visited:
            existing_hop, existing_dist = visited[current_id]
            if existing_hop <= hop_count and existing_dist <= weighted_dist:
                continue
        
        visited[current_id] = (hop_count, weighted_dist)
        
        if current_id not in relationship_map:
            continue
        
        # Explore neighbors
        for target_id, rel_type, weight in relationship_map[current_id]:
            new_hop = hop_count + 1
            new_dist = weighted_dist + (1.0 / weight)  # Inverse weight as distance
            
            if target_id in visited:
                existing_hop, existing_dist = visited[target_id]
                if existing_hop < new_hop or (existing_hop == new_hop and existing_dist <= new_dist):
                    continue
            
            visited[target_id] = (new_hop, new_dist)
            
            # Check if this is a target
            if target_id in target_concept_ids:
                results[target_id] = (new_hop, new_dist)
            
            # Continue traversal
            heapq.heappush(queue, (new_dist, target_id, new_hop))
    
    return results


def calculate_path_density(
    concept_id: int,
    domain_cluster: Set[int],
    relationship_map: Dict[int, List[Tuple[int, str, float]]],
    max_hops: int = 3,
) -> float:
    """
    Calculate path density score for a concept relative to a domain cluster.
    
    Args:
        concept_id: Concept ID to calculate density for
        domain_cluster: Set of concept IDs in domain cluster
        relationship_map: Map from concept_id to relationships
        max_hops: Maximum number of hops
        
    Returns:
        Path density score [0.0, 1.0]
    """
    paths = find_paths_3hop(concept_id, domain_cluster, relationship_map, max_hops)
    
    if len(domain_cluster) == 0:
        return 0.0
    
    total_score = 0.0
    for target_id, hop_count in paths.items():
        # Distance decay: 1 / (hop_count + 1)²
        weight = 1.0 / ((hop_count + 1) ** 2)
        total_score += weight
    
    # Normalize by cluster size
    return total_score / len(domain_cluster)

