"""
Query builder for SurrealQL queries.
Single Responsibility: Build dynamic SurrealQL queries.
"""
from typing import List, Optional, Dict, Any
from models.domain import QueryParameters


class QueryBuilder:
    """
    Builds dynamic SurrealQL queries.
    Follows Open/Closed Principle - extensible for new query types.
    """

    @staticmethod
    def build_select_query(params: QueryParameters) -> str:
        """
        Build a SELECT query with optional filtering.
        
        Args:
            params: Query parameters
            
        Returns:
            SurrealQL query string
        """
        columns_str = ", ".join(params.columns)
        query = f"SELECT {columns_str} FROM {params.table_name}"

        if params.filter_column and params.filter_values:
            # Escape and quote string values
            formatted_values = [
                f"'{v}'" if isinstance(v, str) else str(v) 
                for v in params.filter_values
            ]
            values_str = ", ".join(formatted_values)
            query += f" WHERE {params.filter_column} IN [{values_str}]"

        if params.limit:
            query += f" LIMIT {params.limit}"

        query += ";"
        return query

    @staticmethod
    def build_concept_lookup_query(
        concept_names: List[str],
        table_name: str = "concepts"
    ) -> str:
        """
        Build a query to lookup concept IDs by names.
        
        Args:
            concept_names: List of concept names to lookup
            table_name: Name of the concepts table
            
        Returns:
            SurrealQL query string
        """
        # Remove duplicates and escape quotes
        unique_names = list(set(concept_names))
        escaped_names = [name.replace("'", "\\'") for name in unique_names]
        formatted_names = ", ".join(f"'{name}'" for name in escaped_names)
        
        return f"""
        SELECT concept_name, id 
        FROM {table_name} 
        WHERE concept_name IN [{formatted_names}];
        """

    @staticmethod
    def build_traversal_query(
        source_ids: List[str],
        target_ids: List[str],
        max_depth: int,
        relationship_type: str = "related_to"
    ) -> str:
        """
        Build a graph traversal query to find paths between sources and targets.
        
        Args:
            source_ids: List of source node IDs
            target_ids: List of target node IDs
            max_depth: Maximum traversal depth
            relationship_type: Type of relationship to traverse
            
        Returns:
            SurrealQL query string
        """
        if not source_ids or not target_ids:
            raise ValueError("Source and target IDs cannot be empty")

        source_id_list_str = f"[{', '.join(source_ids)}]"
        target_id_list_str = f"[{', '.join(target_ids)}]"

        depth_queries = []
        for depth in range(1, max_depth + 1):
            target_pointer = QueryBuilder._build_target_pointer(depth, relationship_type)
            triples_block = QueryBuilder._build_triples_block(depth, target_pointer, relationship_type)
            
            block = f"""
                (
                    SELECT 
                        {target_pointer} AS target_id, 
                        {target_pointer}.concept_name AS target_name, 
                        {depth} AS hops,
                        [{triples_block}] AS path_triples
                    FROM {target_id_list_str}<->{relationship_type}
                )
            """
            depth_queries.append(block)

        union_query = ",\n".join(depth_queries)
        
        query = f"""
        SELECT 
            id AS source_id,
            concept_name AS source_name,
            array::flatten([
                {union_query}
            ]) AS found_paths
        FROM {source_id_list_str};
        """
        
        return query

    @staticmethod
    def _build_node_chain(steps: int, relationship_type: str = "related_to") -> str:
        """
        Build a node chain for traversal.
        
        Args:
            steps: Number of traversal steps
            relationship_type: Type of relationship
            
        Returns:
            Chain string like: ->related_to.out->related_to.out
        """
        if steps == 0:
            return ""
        return "".join([f"->{relationship_type}.out" for _ in range(steps)])

    @staticmethod
    def _build_target_pointer(depth: int, relationship_type: str) -> str:
        """Build the target pointer for a given depth."""
        return "out" + QueryBuilder._build_node_chain(depth - 1, relationship_type)

    @staticmethod
    def _build_triples_block(
        depth: int, 
        target_pointer: str, 
        relationship_type: str
    ) -> str:
        """
        Build the triples block for path representation.
        
        Args:
            depth: Traversal depth
            target_pointer: Pointer to target node
            relationship_type: Type of relationship
            
        Returns:
            Triples block string
        """
        triples_objects = []
        
        for step in range(depth):
            # Define relationship pointer
            if step == 0:
                rel_ptr = ""
            else:
                prev_node_chain = "out" + QueryBuilder._build_node_chain(step - 1, relationship_type)
                rel_ptr = f"{prev_node_chain}->{relationship_type}"

            # Build triple components
            from_node = (
                "$parent.concept_name" if step == 0 
                else f"out{QueryBuilder._build_node_chain(step - 1, relationship_type)}.concept_name"
            )
            
            rel_id_acc = (
                "relationship_id" if step == 0 
                else f"{rel_ptr}.relationship_id"
            )
            
            if step == depth - 1:
                to_node = f"{target_pointer}.concept_name"
            elif step == 0:
                to_node = "out.concept_name"
            else:
                to_node = f"{rel_ptr}.out.concept_name"

            triple_str = (
                f"{{ from_name: {from_node}, "
                f"rel_id: {rel_id_acc}, "
                f"to_name: {to_node} }}"
            )
            triples_objects.append(triple_str)

        return ", ".join(triples_objects)

    @staticmethod
    def build_filtered_traversal_query(
        source_ids: List[str],
        target_ids: List[str],
        max_depth: int,
        node_filters: Optional[Dict[str, Any]] = None,
        edge_filters: Optional[Dict[str, Any]] = None,
        relationship_type: str = "related_to"
    ) -> str:
        """
        Build a traversal query with additional filters on nodes and edges.
        
        Args:
            source_ids: Source node IDs
            target_ids: Target node IDs
            max_depth: Maximum depth
            node_filters: Filters for node properties
            edge_filters: Filters for edge properties
            relationship_type: Relationship type
            
        Returns:
            Filtered SurrealQL query string
        """
        # Base query
        base_query = QueryBuilder.build_traversal_query(
            source_ids, target_ids, max_depth, relationship_type
        )
        
        # TODO: Implement filtering logic on metadata
        # This would require modifying the traversal logic to include WHERE clauses
        
        return base_query

    @staticmethod
    def build_count_query(table_name: str, filter_column: Optional[str] = None, 
                         filter_values: Optional[List[Any]] = None) -> str:
        """
        Build a COUNT query.
        
        Args:
            table_name: Table to count from
            filter_column: Optional filter column
            filter_values: Optional filter values
            
        Returns:
            COUNT query string
        """
        query = f"SELECT count() FROM {table_name}"
        
        if filter_column and filter_values:
            formatted_values = [
                f"'{v}'" if isinstance(v, str) else str(v) 
                for v in filter_values
            ]
            values_str = ", ".join(formatted_values)
            query += f" WHERE {filter_column} IN [{values_str}]"
        
        query += ";"
        return query
