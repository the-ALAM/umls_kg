"""
Concept resolution service.
Single Responsibility: Handle concept name to ID resolution.
"""
from typing import List, Dict

from database.client import SurrealClient
from database.queries import QueryBuilder
from processors.data_processor import DataProcessor
from processors.file_processor import FileProcessor
from utils.logger import LoggerFactory
from utils.exceptions import (
    ConceptResolutionError,
    DataValidationError,
    FileProcessingError
)


class ConceptService:
    """
    Service for resolving concept names to IDs.
    Follows Dependency Inversion - depends on abstractions (client interface).
    """

    def __init__(
        self,
        db_client: SurrealClient,
        data_processor: DataProcessor,
        file_processor: FileProcessor
    ):
        """
        Initialize concept service with dependencies.
        
        Args:
            db_client: Database client
            data_processor: Data processor
            file_processor: File processor
        """
        self.db_client = db_client
        self.data_processor = data_processor
        self.file_processor = file_processor
        self.logger = LoggerFactory.get_logger(__name__)

    async def resolve_concept_names_to_ids(
        self,
        concept_names: List[str],
        table_name: str = "denorm_concept_subset_poc"
    ) -> Dict[str, str]:
        """
        Resolve concept names to their database IDs.
        
        Args:
            concept_names: List of concept names to resolve
            table_name: Name of the concepts table
            
        Returns:
            Dictionary mapping concept names to IDs
            
        Raises:
            ConceptResolutionError: If resolution fails
        """
        if not concept_names:
            self.logger.warning("No concept names provided for resolution")
            return {}

        try:
            # Clean names
            cleaned_names = self.data_processor.clean_concept_names(concept_names)
            unique_names = list(set(cleaned_names))

            self.logger.info(f"Resolving {len(unique_names)} unique concept names to IDs")

            # Build query using QueryBuilder
            query = QueryBuilder.build_concept_lookup_query(unique_names, table_name)

            # Execute query
            response = await self.db_client.execute_query(query)

            # Parse response into name->ID mapping
            id_map = self.data_processor.construct_id_name_map(
                response,
                name_key="concept_name",
                id_key="id"
            )

            self.logger.info(f"Successfully resolved {len(id_map)}/{len(unique_names)} concept names")
            return id_map

        except Exception as e:
            self.logger.error(f"Failed to resolve concept names: {e}")
            raise ConceptResolutionError(f"Concept resolution failed: {e}") from e

    def load_source_concepts_from_csv(self, csv_path: str) -> List[str]:
        """
        Load source concepts from a CSV file.

        Args:
            csv_path: Path to CSV file containing source concepts

        Returns:
            List of source concept names

        Raises:
            FileProcessingError: If file cannot be read
            DataValidationError: If CSV format is invalid
        """
        try:
            df = self.file_processor.read_csv(csv_path)

            # Expect a column named 'concept_name' or similar
            if 'concept_name' in df.columns:
                concepts = self.data_processor.extract_unique_values(df, 'concept_name')
            elif 'ConceptName' in df.columns:
                concepts = self.data_processor.extract_unique_values(df, 'ConceptName')
            elif len(df.columns) > 0:
                # Use first column as fallback
                concepts = self.data_processor.extract_unique_values(df, df.columns[0])
            else:
                raise DataValidationError("CSV file has no columns")

            self.logger.info(f"Loaded {len(concepts)} source concepts from {csv_path}")
            return concepts

        except FileProcessingError:
            raise
        except Exception as e:
            self.logger.error(f"Failed to load source concepts from CSV: {e}")
            raise DataValidationError(f"Failed to load source concepts: {e}") from e

    def load_target_concepts_from_csv(
        self,
        csv_path: str,
        drg_codes: List[int]
    ) -> List[str]:
        """
        Load target concepts from CSV filtered by DRG codes.

        Args:
            csv_path: Path to CSV file containing target concepts
            drg_codes: List of DRG codes to filter by

        Returns:
            List of target concept names

        Raises:
            FileProcessingError: If file cannot be read
            DataValidationError: If CSV format is invalid
        """
        try:
            df = self.file_processor.read_csv(csv_path)

            # Validate required columns
            required_columns = ['DRG', 'DiagnosisDescription']
            self.data_processor.validate_dataframe_columns(df, required_columns)

            # Filter by DRG codes
            filtered_df = self.data_processor.filter_dataframe(df, 'DRG', drg_codes)

            if filtered_df.empty:
                self.logger.warning(f"No records found for DRG codes: {drg_codes}")
                return []

            # Extract unique diagnosis descriptions
            concepts = self.data_processor.extract_unique_values(
                filtered_df,
                'DiagnosisDescription'
            )

            self.logger.info(
                f"Loaded {len(concepts)} target concepts from {csv_path} "
                f"(filtered by DRG codes: {drg_codes})"
            )
            return concepts

        except FileProcessingError:
            raise
        except Exception as e:
            self.logger.error(f"Failed to load target concepts from CSV: {e}")
            raise DataValidationError(f"Failed to load target concepts: {e}") from e

    def get_concept_ids(
        self,
        concept_names: List[str],
        id_map: Dict[str, str]
    ) -> List[str]:
        """
        Extract concept IDs from an ID mapping for given concept names.

        Args:
            concept_names: List of concept names to get IDs for
            id_map: Dictionary mapping concept names to IDs

        Returns:
            List of concept IDs (only for names that exist in the map)
        """
        ids = [id_map[name] for name in concept_names if name in id_map]
        self.logger.debug(
            f"Extracted {len(ids)}/{len(concept_names)} IDs from mapping"
        )
        return ids
