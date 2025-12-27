"""
Domain models and data structures.
Single Responsibility: Define data structures used across the application.
"""
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional


@dataclass
class PathTriple:
    """Represents a single triple in a path: (from)-[relationship]->(to)"""
    from_name: str
    rel_id: str
    to_name: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:
        return f"('{self.from_name}')->('{self.rel_id}')->('{self.to_name}')"


@dataclass
class GraphPath:
    """Represents a complete path through the graph."""
    source_name: str
    target_name: str
    hops: int
    triples: List[PathTriple]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'source_name': self.source_name,
            'target_name': self.target_name,
            'hops': self.hops,
            'triples': [t.to_dict() for t in self.triples]
        }

    def format_triples(self) -> str:
        """Format triples as a readable string."""
        if not self.triples:
            return "[]"
        return f"[{', '.join(str(t) for t in self.triples)}]"


@dataclass
class TraversalResult:
    """
    Result of a graph traversal between source and target.
    Contains all paths found and statistics.
    """
    source_name: str
    target_name: str
    min_hops: int
    max_hops: int
    shortest_path_triples: str
    all_paths_triples: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_paths(
        cls, 
        source_name: str, 
        target_name: str, 
        paths: List[GraphPath]
    ) -> 'TraversalResult':
        """Create a TraversalResult from a list of GraphPath objects."""
        if not paths:
            return cls(
                source_name=source_name,
                target_name=target_name,
                min_hops=0,
                max_hops=0,
                shortest_path_triples="[]",
                all_paths_triples="[]"
            )

        hops_list = [p.hops for p in paths]
        min_hops = min(hops_list)
        max_hops = max(hops_list)

        # Get shortest path(s)
        shortest_paths = [p for p in paths if p.hops == min_hops]
        shortest_str = shortest_paths[0].format_triples() if shortest_paths else "[]"

        # Combine all paths
        all_paths_str = f"[{', '.join(p.format_triples() for p in paths)}]"

        return cls(
            source_name=source_name,
            target_name=target_name,
            min_hops=min_hops,
            max_hops=max_hops,
            shortest_path_triples=shortest_str,
            all_paths_triples=all_paths_str
        )


@dataclass
class Concept:
    """Represents a concept node in the knowledge graph."""
    id: str
    name: str
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QueryParameters:
    """Parameters for database queries."""
    table_name: str
    columns: List[str]
    filter_column: Optional[str] = None
    filter_values: Optional[List[str]] = None
    limit: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}
