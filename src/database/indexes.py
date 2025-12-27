"""Index creation and management for SurrealDB."""

from typing import List, Optional
from src.database.surreal_client import SurrealDBSync


class IndexManager:
    """Manage SurrealDB indexes for optimization."""
    
    def __init__(self, client: Optional[SurrealDBSync] = None):
        """
        Initialize index manager.
        
        Args:
            client: SurrealDB client (creates new if None)
        """
        self.client = client or SurrealDBSync()
        if not self.client._connected:
            self.client.connect()
    
    def create_vector_index(
        self,
        table: str,
        field: str,
        dimension: int = 1536,
    ) -> None:
        """
        Create vector index (MTREE) for semantic embeddings.
        
        Args:
            table: Table name
            field: Field name containing vectors
            dimension: Vector dimension
        """
        query = f"""
        DEFINE INDEX idx_{field}_vector ON {table} FIELDS {field} MTREE DIMENSION {dimension}
        """
        self.client.query(query)
    
    def create_graph_index(
        self,
        table: str,
        field: str,
    ) -> None:
        """
        Create graph index for relationship traversal.
        
        Args:
            table: Table name
            field: Field name to index
        """
        query = f"""
        DEFINE INDEX idx_{table}_{field} ON {table} FIELDS {field}
        """
        self.client.query(query)
    
    def create_composite_index(
        self,
        table: str,
        fields: List[str],
        index_name: Optional[str] = None,
    ) -> None:
        """
        Create composite index for multiple fields.
        
        Args:
            table: Table name
            fields: List of field names
            index_name: Optional custom index name
        """
        if index_name is None:
            index_name = f"idx_{table}_{'_'.join(fields)}"
        
        fields_str = ", ".join(fields)
        query = f"""
        DEFINE INDEX {index_name} ON {table} FIELDS {fields_str}
        """
        self.client.query(query)
    
    def create_all_indexes(self) -> None:
        """Create all recommended indexes."""
        # Vector index for semantic embeddings
        self.create_vector_index("metric", "semantic_embed", dimension=1536)
        
        # Graph indexes for relationships
        self.create_graph_index("relates_to", "relationship_id")
        self.create_graph_index("relates_to", "category")
        self.create_graph_index("is_ancestor_of", "min_levels_of_separation")
        
        # Composite indexes for common queries
        self.create_composite_index("relevance_score", ["domain_cluster_id", "score"])
        self.create_composite_index("metric", ["concept_id", "model_version"])
        
        # Single field indexes
        self.create_graph_index("concept", "vocabulary_id")
        self.create_graph_index("concept", "domain_id")
        self.create_graph_index("relevance_score", "concept_id")
        self.create_graph_index("relevance_score", "domain_cluster_id")
    
    def drop_index(self, index_name: str) -> None:
        """
        Drop an index.
        
        Args:
            index_name: Index name to drop
        """
        query = f"REMOVE INDEX {index_name}"
        self.client.query(query)
    
    def list_indexes(self, table: Optional[str] = None) -> List[dict]:
        """
        List all indexes, optionally filtered by table.
        
        Args:
            table: Optional table name to filter by
            
        Returns:
            List of index information
        """
        if table:
            query = f"INFO FOR TABLE {table}"
        else:
            query = "INFO FOR DB"
        
        result = self.client.query(query)
        return result if result else []


def create_optimized_indexes(client: Optional[SurrealDBSync] = None) -> None:
    """
    Create all optimized indexes for the UMLS knowledge graph.
    
    Args:
        client: SurrealDB client (creates new if None)
    """
    manager = IndexManager(client)
    manager.create_all_indexes()

