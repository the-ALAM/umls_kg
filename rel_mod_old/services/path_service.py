"""
Path finding and processing service.
Single Responsibility: Handle graph traversal and path processing.
"""
from typing import List, Dict, Any
from database.client import SurrealClient
from database.queries import QueryBuilder
from processors.data_processor import DataProcessor
from processors.file_processor import FileProcessor
from models.domain import TraversalResult
from utils.logger import LoggerFactory
from utils.exceptions import QueryExecutionError, FileProcessingError, DataValidationError


class PathService:
    """
    Service for finding and processing graph paths.
    Follows Dependency Inversion - depends on abstractions (client interface).
    """

    def __init__(
        self,
        db_client: SurrealClient,
        data_processor: DataProcessor,
        file_processor: FileProcessor
    ):
        """
        Initialize path service with dependencies.

        Args:
            db_client: Database client
            data_processor: Data processor
            file_processor: File processor
        """
        self.db_client = db_client
        self.data_processor = data_processor
        self.file_processor = file_processor
        self.logger = LoggerFactory.get_logger(__name__)

    async def find_and_process_paths(
        self,
        source_ids: List[str],
        target_ids: List[str],
        max_depth: int,
        relationship_type: str = "related_to"
    ) -> List[TraversalResult]:
        """
        Find paths between source and target nodes and process results.

        Args:
            source_ids: List of source node IDs
            target_ids: List of target node IDs
            max_depth: Maximum traversal depth
            relationship_type: Type of relationship to traverse

        Returns:
            List of TraversalResult objects

        Raises:
            QueryExecutionError: If traversal query fails
            DataValidationError: If results cannot be parsed
        """
        if not source_ids or not target_ids:
            self.logger.warning("Empty source or target IDs provided")
            return []

        try:
            self.logger.info(
                f"Finding paths: {len(source_ids)} sources -> "
                f"{len(target_ids)} targets (max depth: {max_depth})"
            )

            # Build traversal query
            query = QueryBuilder.build_traversal_query(
                source_ids,
                target_ids,
                max_depth,
                relationship_type
            )

            # Execute query
            raw_results = await self.db_client.execute_query(query)

            if not raw_results:
                self.logger.info("No paths found between sources and targets")
                return []

            # Parse results into TraversalResult objects
            parsed_results = self.data_processor.parse_traversal_results(raw_results)

            self.logger.info(
                f"Found {len(parsed_results)} paths between sources and targets"
            )
            return parsed_results

        except QueryExecutionError:
            raise
        except Exception as e:
            self.logger.error(f"Failed to find and process paths: {e}")
            raise QueryExecutionError(f"Path finding failed: {e}") from e

    def save_results_to_csv(
        self,
        results: List[TraversalResult],
        filepath: str
    ) -> None:
        """
        Save traversal results to CSV file.

        Args:
            results: List of TraversalResult objects
            filepath: Output file path

        Raises:
            FileProcessingError: If file cannot be written
        """
        if not results:
            self.logger.warning("No results to save")
            return

        try:
            # Convert results to DataFrame
            df = self.data_processor.results_to_dataframe(results)

            # Write to CSV
            self.file_processor.write_csv(df, filepath, index=False)

            self.logger.info(
                f"Successfully saved {len(results)} results to {filepath}"
            )

        except FileProcessingError:
            raise
        except Exception as e:
            self.logger.error(f"Failed to save results to CSV: {e}")
            raise FileProcessingError(f"Failed to save results: {e}") from e

    def get_statistics(
        self,
        results: List[TraversalResult]
    ) -> Dict[str, Any]:
        """
        Calculate statistics from traversal results.

        Args:
            results: List of TraversalResult objects

        Returns:
            Dictionary with statistics:
            - total_paths: Total number of paths found
            - unique_sources: Number of unique source concepts
            - unique_targets: Number of unique target concepts
            - min_hops: Minimum number of hops
            - max_hops: Maximum number of hops
            - avg_hops: Average number of hops
        """
        if not results:
            return {
                'total_paths': 0,
                'unique_sources': 0,
                'unique_targets': 0,
                'min_hops': 0,
                'max_hops': 0,
                'avg_hops': 0.0
            }

        total_paths = len(results)

        # Handle source_name which might be a list or a single string
        all_sources = set()
        for r in results:
            if isinstance(r.source_name, list):
                all_sources.update(r.source_name)
            else:
                all_sources.add(r.source_name)
        unique_sources = len(all_sources)

        # Handle target_name which might be a list or a single string
        all_targets = set()
        # Helper function to flatten potentially nested lists of strings, yielding only strings
        def _flatten_strings(data):
            if isinstance(data, list):
                for item in data:
                    yield from _flatten_strings(item)
            elif isinstance(data, str):
                yield data

        for r in results:
            # Use the helper to ensure only individual strings are added to the set
            all_targets.update(_flatten_strings(r.target_name))
        unique_targets = len(all_targets)

        # Calculate hop statistics
        hops_list = []
        for result in results:
            # Use min_hops for statistics (could also use max_hops)
            hops_list.append(result.min_hops)

        min_hops = min(hops_list) if hops_list else 0
        max_hops = max(hops_list) if hops_list else 0
        avg_hops = sum(hops_list) / len(hops_list) if hops_list else 0.0

        stats = {
            'total_paths': total_paths,
            'unique_sources': unique_sources,
            'unique_targets': unique_targets,
            'min_hops': min_hops,
            'max_hops': max_hops,
            'avg_hops': avg_hops
        }

        self.logger.debug(f"Calculated statistics: {stats}")
        return stats


    def aggregate_paths_by_source_target(
        self,
        results: List[TraversalResult]
    ) -> Dict[tuple[str, str], List[TraversalResult]]:
        """
        Aggregate results by source-target pairs.

        Args:
            results: List of TraversalResult objects

        Returns:
            Dictionary mapping (source_name, target_name) tuples to lists of results
        """
        aggregated = self.data_processor.aggregate_paths_by_source_target(results)
        self.logger.info(
            f"Aggregated {len(results)} results into "
            f"{len(aggregated)} source-target pairs"
        )
        return aggregated

