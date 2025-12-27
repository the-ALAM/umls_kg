"""
SurrealDB client interface.
Single Responsibility: Handle database connections and query execution.
"""
import asyncio
from typing import Any, Dict, List, Optional
from surrealdb import Surreal

from config.settings import DatabaseConfig
from utils.logger import LoggerFactory
from utils.exceptions import DatabaseConnectionError, QueryExecutionError


class SurrealClient:
    """
    Database client for SurrealDB interactions.
    Follows Interface Segregation - provides only database operations.
    """

    def __init__(self, config: DatabaseConfig):
        """
        Initialize the SurrealDB client.
        
        Args:
            config: Database configuration
        """
        self.config = config
        self.db: Optional[Surreal] = None
        self.logger = LoggerFactory.get_logger(__name__)
        self._connected = False

    def connect(self) -> None:
        """
        Establish connection to SurrealDB.
        
        Raises:
            DatabaseConnectionError: If connection fails
        """
        try:
            self.db = Surreal(self.config.url)
            self.db.signin({
                "username": self.config.username,
                "password": self.config.password
            })
            self.db.use(self.config.namespace, self.config.database)
            self._connected = True
            self.logger.info(
                f"✅ Connected to SurrealDB: {self.config.namespace}/{self.config.database}"
            )
        except Exception as e:
            self.logger.error(f"Failed to connect to database: {e}")
            raise DatabaseConnectionError(f"Connection failed: {e}") from e

    def close(self) -> None:
        """Close the database connection."""
        if self.db and self._connected:
            try:
                self.db.close()
                self._connected = False
                self.logger.info("Database connection closed")
            except Exception as e:
                self.logger.warning(f"Error closing database connection: {e}")

    def is_connected(self) -> bool:
        """Check if the client is connected to the database."""
        return self._connected

    async def execute_query(
        self, 
        query: str, 
        params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Execute a SurrealQL query.
        
        Args:
            query: SurrealQL query string
            params: Optional query parameters
            
        Returns:
            Query results
            
        Raises:
            DatabaseConnectionError: If not connected
            QueryExecutionError: If query execution fails
        """
        if not self._connected or not self.db:
            raise DatabaseConnectionError("Not connected to database")

        try:
            params = params or {}
            response = await asyncio.to_thread(self.db.query, query, params)
            
            # Handle different response formats
            if isinstance(response, list):
                if response and isinstance(response[0], dict) and 'result' in response[0]:
                    return response[0].get('result', [])
                return response
            return response
            
        except Exception as e:
            self.logger.error(f"Query execution failed: {e}")
            raise QueryExecutionError(f"Query failed: {e}") from e

    async def execute_batch(
        self, 
        queries: List[tuple[str, Dict[str, Any]]]
    ) -> List[Any]:
        """
        Execute multiple queries in sequence.
        
        Args:
            queries: List of (query, params) tuples
            
        Returns:
            List of query results
        """
        results = []
        for query, params in queries:
            result = await self.execute_query(query, params)
            results.append(result)
        return results

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    async def __aenter__(self):
        """Async context manager entry."""
        self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        self.close()
