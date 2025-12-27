"""
Main application entry point.
Single Responsibility: Orchestrate the complete pipeline.
"""
import asyncio
import sys
from typing import Optional

from config.settings import ApplicationConfig, ConfigLoader
from database.client import SurrealClient
from processors.file_processor import FileProcessor
from processors.data_processor import DataProcessor
from services.concept_service import ConceptService
from services.path_service import PathService
from utils.logger import LoggerFactory
from utils.exceptions import GraphTraversalException


class GraphTraversalPipeline:
    """
    Main pipeline orchestrator.
    Follows Dependency Injection and coordinates all services.
    """

    def __init__(self, config: ApplicationConfig):
        """
        Initialize pipeline with configuration.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.logger = LoggerFactory.get_logger(__name__)

        # Initialize components
        self.db_client = SurrealClient(config.database)
        self.file_processor = FileProcessor()
        self.data_processor = DataProcessor()
        
        # Initialize services with dependencies
        self.concept_service = ConceptService(
            self.db_client,
            self.data_processor,
            self.file_processor
        )
        self.path_service = PathService(
            self.db_client,
            self.data_processor,
            self.file_processor
        )

    async def run(self) -> bool:
        """
        Execute the complete graph traversal pipeline.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info("=" * 60)
            self.logger.info("Starting Graph Traversal Pipeline")
            self.logger.info("=" * 60)

            # Step 1: Load input data
            self.logger.info("\n--- Step 1: Loading Input Data ---")
            source_concepts, target_concepts = await self._load_input_data()
            
            if not source_concepts or not target_concepts:
                self.logger.error("Failed to load input data")
                return False

            # Step 2: Connect to database
            self.logger.info("\n--- Step 2: Connecting to Database ---")
            self.db_client.connect()

            # Step 3: Resolve concept names to IDs
            self.logger.info("\n--- Step 3: Resolving Concept IDs ---")
            source_ids, target_ids = await self._resolve_concepts(
                source_concepts, 
                target_concepts
            )

            if not source_ids or not target_ids:
                self.logger.error("Failed to resolve concepts to IDs")
                return False

            # Step 4: Execute graph traversal
            self.logger.info(
                f"\n--- Step 4: Executing Graph Traversal "
                f"(max depth: {self.config.traversal.max_depth}) ---"
            )
            results = await self._execute_traversal(source_ids, target_ids)

            if not results:
                self.logger.warning("No paths found between sources and targets")
                return True  # Not an error, just no paths

            # Step 5: Save results
            self.logger.info("\n--- Step 5: Saving Results ---")
            await self._save_results(results)

            # Step 6: Display statistics
            self.logger.info("\n--- Step 6: Results Summary ---")
            self._display_statistics(results)
            #TODO - save the statistics to a file

            self.logger.info("\n" + "=" * 60)
            self.logger.info("Pipeline completed successfully!")
            self.logger.info("=" * 60)
            return True

        except GraphTraversalException as e:
            self.logger.error(f"Pipeline failed: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}", exc_info=True)
            return False
        finally:
            self.db_client.close()

    async def _load_input_data(self) -> tuple[list[str], list[str]]:
        """Load source and target concepts from configured sources."""

        # Load source concepts
        if self.config.paths.source_csv_path:
            source_concepts = self.concept_service.load_source_concepts_from_csv(
                self.config.paths.source_csv_path
            )
        else:
            source_concepts = self.config.traversal.source_concepts
            self.logger.info(f"Using configured source concepts: {len(source_concepts)}")

        # Load target concepts from CSV filtered by DRG codes
        target_concepts = self.concept_service.load_target_concepts_from_csv(
            self.config.paths.target_csv_path,
            self.config.traversal.target_drg_codes
        )

        self.logger.info(f"Loaded {len(source_concepts)} source concepts")
        self.logger.info(f"Loaded {len(target_concepts)} target concepts")

        return source_concepts, target_concepts

    async def _resolve_concepts(
        self,
        source_concepts: list[str],
        target_concepts: list[str]
    ) -> tuple[list[str], list[str]]:
        """Resolve concept names to database IDs."""
        
        # Combine all concepts for batch resolution
        all_concepts = source_concepts + target_concepts
        self.logger.info(f"Resolving {len(all_concepts)} unique concepts to IDs")

        # Resolve all at once
        id_map = await self.concept_service.resolve_concept_names_to_ids(all_concepts)

        # Extract IDs for sources and targets
        source_ids = self.concept_service.get_concept_ids(source_concepts, id_map)
        target_ids = self.concept_service.get_concept_ids(target_concepts, id_map)

        self.logger.info(f"Resolved {len(source_ids)}/{len(source_concepts)} source IDs")
        self.logger.info(f"Resolved {len(target_ids)}/{len(target_concepts)} target IDs")

        return source_ids, target_ids

    async def _execute_traversal(
        self,
        source_ids: list[str],
        target_ids: list[str]
    ):
        """Execute graph traversal to find paths."""
        
        results = await self.path_service.find_and_process_paths(
            source_ids,
            target_ids,
            self.config.traversal.max_depth
        )

        self.logger.info(f"Found {len(results)} path results")
        return results

    async def _save_results(self, results):
        """Save results to configured output paths."""
        
        # Save to CSV
        self.path_service.save_results_to_csv(
            results,
            self.config.paths.output_csv_path
        )

    def _display_statistics(self, results):
        """Display summary statistics."""
        
        stats = self.path_service.get_statistics(results)
        
        self.logger.info(f"Total paths found: {stats['total_paths']}")
        self.logger.info(f"Unique sources: {stats['unique_sources']}")
        self.logger.info(f"Unique targets: {stats['unique_targets']}")
        self.logger.info(f"Hop range: {stats['min_hops']} - {stats['max_hops']}")
        self.logger.info(f"Average hops: {stats['avg_hops']:.2f}")


async def main():
    """Main entry point."""
    try:
        # Load configuration
        # You can use create_default() or create_from_environment()
        config = ApplicationConfig.create_default()

        # Or load from environment:
        # config = ApplicationConfig.create_from_environment()
        
        # Create and run pipeline
        pipeline = GraphTraversalPipeline(config)
        success = await pipeline.run()
        
        sys.exit(0 if success else 1)
        
    except Exception as e:
        logger = LoggerFactory.get_logger(__name__)
        logger.error(f"Application failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
