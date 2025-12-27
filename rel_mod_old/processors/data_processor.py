"""
Data transformation and processing.
Single Responsibility: Transform and manipulate data structures.
"""
from typing import List, Dict, Any, Tuple
import pandas as pd

from models.domain import TraversalResult, PathTriple, GraphPath
from utils.logger import LoggerFactory
from utils.exceptions import DataValidationError


class DataProcessor:
    """
    Handles data transformation and processing.
    Agnostic to data source - works with generic data structures.
    """

    def __init__(self):
        self.logger = LoggerFactory.get_logger(__name__)

    def filter_dataframe(
        self,
        df: pd.DataFrame,
        column: str,
        values: List[Any]
    ) -> pd.DataFrame:
        """
        Filter DataFrame by column values.
        
        Args:
            df: Input DataFrame
            column: Column to filter on
            values: Values to filter for
            
        Returns:
            Filtered DataFrame
            
        Raises:
            DataValidationError: If column doesn't exist
        """
        try:
            if column not in df.columns:
                raise DataValidationError(f"Column '{column}' not found in DataFrame")
            
            filtered = df[df[column].isin(values)]
            self.logger.info(
                f"Filtered DataFrame: {len(filtered)}/{len(df)} rows match criteria"
            )
            return filtered
            
        except KeyError as e:
            raise DataValidationError(f"Invalid column: {e}") from e

    def extract_unique_values(
        self,
        df: pd.DataFrame,
        column: str,
        dropna: bool = True
    ) -> List[Any]:
        """
        Extract unique values from a DataFrame column.
        
        Args:
            df: Input DataFrame
            column: Column name
            dropna: Whether to drop NA values
            
        Returns:
            List of unique values
        """
        try:
            if column not in df.columns:
                raise DataValidationError(f"Column '{column}' not found")
            
            series = df[column]
            if dropna:
                series = series.dropna()
            
            unique_values = series.unique().tolist()
            self.logger.info(f"Extracted {len(unique_values)} unique values from '{column}'")
            return unique_values
            
        except Exception as e:
            self.logger.error(f"Error extracting unique values: {e}")
            raise DataValidationError(f"Failed to extract values: {e}") from e

    def construct_id_name_map(
        self,
        data: List[Dict[str, Any]],
        name_key: str = "concept_name",
        id_key: str = "id"
    ) -> Dict[str, str]:
        """
        Construct a mapping from names to IDs.
        
        Args:
            data: List of dictionaries containing name and ID
            name_key: Key for name field
            id_key: Key for ID field
            
        Returns:
            Dictionary mapping names to IDs
        """
        lookup = {}
        
        for item in data:
            if not isinstance(item, dict):
                continue
                
            name = item.get(name_key)
            record_id = item.get(id_key)
            
            if name and record_id:
                # Handle SurrealDB RecordID objects
                id_str = str(record_id) if not isinstance(record_id, str) else record_id
                lookup[name] = id_str
        
        self.logger.info(f"Constructed ID mapping for {len(lookup)} items")
        return lookup

    def format_triple_string(self, triples_list: List[Dict]) -> str:
        """
        Convert list of triple dictionaries to readable string.
        Handles both single values and lists for rel_id and to_name.
        
        Args:
            triples_list: List of triple dictionaries
            
        Returns:
            Formatted string representation
        """
        if not triples_list:
            return "[]"

        formatted_triples = []
        
        for step in triples_list:
            from_name = step.get('from_name', 'Unknown')
            rel_id_val = step.get('rel_id', 'Unknown')
            to_name_val = step.get('to_name', 'Unknown')

            # Handle list values for rel_id
            if isinstance(rel_id_val, list):
                if isinstance(to_name_val, list):
                    # Both are lists - iterate by index
                    for i in range(min(len(rel_id_val), len(to_name_val))):
                        rel = rel_id_val[i]
                        to_name = to_name_val[i]
                        formatted_triple = f"('{from_name}')->('{rel}')->('{to_name}')"
                        formatted_triples.append(formatted_triple)
                else:
                    # Only rel_id is list
                    for rel in rel_id_val:
                        formatted_triple = f"('{from_name}')->('{rel}')->('{to_name_val}')"
                        formatted_triples.append(formatted_triple)
            else:
                # Single values
                current_to_name = (
                    to_name_val[0] if isinstance(to_name_val, list) and to_name_val 
                    else to_name_val
                )
                formatted_triple = f"('{from_name}')->('{rel_id_val}')->('{current_to_name}')"
                formatted_triples.append(formatted_triple)
        
        return f"[{', '.join(formatted_triples)}]"

    def parse_traversal_results(
        self,
        results: List[Dict[str, Any]]
    ) -> List[TraversalResult]:
        """
        Parse raw database results into TraversalResult objects.
        
        Args:
            results: Raw results from database query
            
        Returns:
            List of TraversalResult objects
        """
        parsed_rows = []
        
        try:
            for source_record in results:
                src_name = source_record.get('source_name', 'Unknown')
                found_paths_array = source_record.get('found_paths', [])
                
                for path_info in found_paths_array:
                    # Extract target name (handle list format)
                    tgt_name_raw = path_info.get('target_name', ['Unknown'])
                    tgt_name = tgt_name_raw[0] if isinstance(tgt_name_raw, list) else tgt_name_raw
                    
                    hops = path_info.get('hops', 0)
                    
                    # Format path triples
                    shortest_raw = path_info.get('path_triples', [])
                    shortest_str = self.format_triple_string(shortest_raw)
                    
                    # For now, all_paths is same as shortest (single path per result)
                    all_paths_str = shortest_str
                    
                    row = TraversalResult(
                        source_name=src_name,
                        target_name=tgt_name,
                        min_hops=hops,
                        max_hops=hops,
                        shortest_path_triples=shortest_str,
                        all_paths_triples=all_paths_str
                    )
                    parsed_rows.append(row)
            
            self.logger.info(f"Parsed {len(parsed_rows)} traversal results")
            return parsed_rows
            
        except Exception as e:
            self.logger.error(f"Error parsing traversal results: {e}")
            raise DataValidationError(f"Failed to parse results: {e}") from e

    def results_to_dataframe(
        self,
        results: List[TraversalResult]
    ) -> pd.DataFrame:
        """
        Convert TraversalResult objects to DataFrame.
        
        Args:
            results: List of TraversalResult objects
            
        Returns:
            DataFrame with results
        """
        if not results:
            self.logger.warning("No results to convert to DataFrame")
            return pd.DataFrame()
        
        data = [result.to_dict() for result in results]
        df = pd.DataFrame(data)
        
        self.logger.info(f"Converted {len(results)} results to DataFrame")
        return df

    def aggregate_paths_by_source_target(
        self,
        results: List[TraversalResult]
    ) -> Dict[Tuple[str, str], List[TraversalResult]]:
        """
        Group results by source-target pairs.
        
        Args:
            results: List of TraversalResult objects
            
        Returns:
            Dictionary mapping (source, target) to list of results
        """
        aggregated = {}
        
        for result in results:
            key = (result.source_name, result.target_name)
            if key not in aggregated:
                aggregated[key] = []
            aggregated[key].append(result)
        
        self.logger.info(f"Aggregated results into {len(aggregated)} source-target pairs")
        return aggregated

    def validate_dataframe_columns(
        self,
        df: pd.DataFrame,
        required_columns: List[str]
    ) -> bool:
        """
        Validate that DataFrame has required columns.
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            
        Returns:
            True if valid
            
        Raises:
            DataValidationError: If columns are missing
        """
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise DataValidationError(
                f"DataFrame missing required columns: {missing}"
            )
        return True

    def clean_concept_names(self, names: List[str]) -> List[str]:
        """
        Clean and standardize concept names.
        
        Args:
            names: List of concept names
            
        Returns:
            Cleaned concept names
        """
        cleaned = []
        for name in names:
            if not name or not isinstance(name, str):
                continue
            # Strip whitespace and normalize
            cleaned_name = name.strip()
            if cleaned_name:
                cleaned.append(cleaned_name)
        
        self.logger.info(f"Cleaned {len(cleaned)}/{len(names)} concept names")
        return cleaned
