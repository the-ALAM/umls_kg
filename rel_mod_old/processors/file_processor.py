"""
File I/O operations.
Single Responsibility: Handle all file reading and writing operations.
"""
import json
from pathlib import Path
from typing import Any, List, Dict
import pandas as pd

from utils.logger import LoggerFactory
from utils.exceptions import FileProcessingError


class FileProcessor:
    """
    Handles file input/output operations.
    Follows Single Responsibility Principle.
    """

    def __init__(self):
        self.logger = LoggerFactory.get_logger(__name__)

    def read_csv(
        self, 
        filepath: str, 
        encoding: str = 'utf-8',
        **kwargs
    ) -> pd.DataFrame:
        """
        Read CSV file into DataFrame.
        
        Args:
            filepath: Path to CSV file
            encoding: File encoding
            **kwargs: Additional pandas read_csv arguments
            
        Returns:
            DataFrame with CSV data
            
        Raises:
            FileProcessingError: If file cannot be read
        """
        try:
            path = Path(filepath)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {filepath}")
            
            self.logger.info(f"Reading CSV: {filepath}")
            df = pd.read_csv(filepath, encoding=encoding, **kwargs)
            
            # Strip whitespace from column names
            df.columns = df.columns.str.strip()
            
            self.logger.info(f"✅ Loaded CSV with {len(df)} rows, {len(df.columns)} columns")
            return df
            
        except FileNotFoundError as e:
            self.logger.error(f"File not found: {filepath}")
            raise FileProcessingError(f"File not found: {filepath}") from e
        except Exception as e:
            self.logger.error(f"Error reading CSV {filepath}: {e}")
            raise FileProcessingError(f"Failed to read CSV: {e}") from e

    def write_csv(
        self, 
        df: pd.DataFrame, 
        filepath: str, 
        index: bool = False,
        encoding: str = 'utf-8',
        **kwargs
    ) -> None:
        """
        Write DataFrame to CSV file.
        
        Args:
            df: DataFrame to write
            filepath: Output file path
            index: Whether to write index
            encoding: File encoding
            **kwargs: Additional pandas to_csv arguments
            
        Raises:
            FileProcessingError: If file cannot be written
        """
        try:
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"Writing CSV: {filepath}")
            df.to_csv(filepath, index=index, encoding=encoding, **kwargs)
            self.logger.info(f"✅ Wrote {len(df)} rows to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error writing CSV {filepath}: {e}")
            raise FileProcessingError(f"Failed to write CSV: {e}") from e

    def read_json(self, filepath: str, encoding: str = 'utf-8') -> Any:
        """
        Read JSON file.
        
        Args:
            filepath: Path to JSON file
            encoding: File encoding
            
        Returns:
            Parsed JSON data
            
        Raises:
            FileProcessingError: If file cannot be read or parsed
        """
        try:
            path = Path(filepath)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {filepath}")
            
            self.logger.info(f"Reading JSON: {filepath}")
            with open(filepath, 'r', encoding=encoding) as f:
                data = json.load(f)
            
            self.logger.info(f"✅ Loaded JSON from {filepath}")
            return data
            
        except FileNotFoundError as e:
            self.logger.error(f"File not found: {filepath}")
            raise FileProcessingError(f"File not found: {filepath}") from e
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in {filepath}: {e}")
            raise FileProcessingError(f"Invalid JSON: {e}") from e
        except Exception as e:
            self.logger.error(f"Error reading JSON {filepath}: {e}")
            raise FileProcessingError(f"Failed to read JSON: {e}") from e

    def write_json(
        self, 
        data: Any, 
        filepath: str, 
        indent: int = 4,
        encoding: str = 'utf-8',
        ensure_ascii: bool = False
    ) -> None:
        """
        Write data to JSON file.
        
        Args:
            data: Data to serialize
            filepath: Output file path
            indent: JSON indentation
            encoding: File encoding
            ensure_ascii: Whether to escape non-ASCII characters
            
        Raises:
            FileProcessingError: If file cannot be written
        """
        try:
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"Writing JSON: {filepath}")
            with open(filepath, 'w', encoding=encoding) as f:
                json.dump(data, f, ensure_ascii=ensure_ascii, indent=indent)
            
            self.logger.info(f"✅ Wrote JSON to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error writing JSON {filepath}: {e}")
            raise FileProcessingError(f"Failed to write JSON: {e}") from e

    def file_exists(self, filepath: str) -> bool:
        """Check if file exists."""
        return Path(filepath).exists()

    def create_directory(self, dirpath: str) -> None:
        """
        Create directory if it doesn't exist.
        
        Args:
            dirpath: Directory path to create
        """
        try:
            Path(dirpath).mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Directory ensured: {dirpath}")
        except Exception as e:
            self.logger.error(f"Error creating directory {dirpath}: {e}")
            raise FileProcessingError(f"Failed to create directory: {e}") from e

    def list_files(self, dirpath: str, pattern: str = "*") -> List[Path]:
        """
        List files in directory matching pattern.
        
        Args:
            dirpath: Directory path
            pattern: Glob pattern
            
        Returns:
            List of file paths
        """
        try:
            path = Path(dirpath)
            if not path.exists():
                return []
            return list(path.glob(pattern))
        except Exception as e:
            self.logger.error(f"Error listing files in {dirpath}: {e}")
            raise FileProcessingError(f"Failed to list files: {e}") from e
