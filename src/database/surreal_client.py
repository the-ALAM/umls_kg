"""
SurrealDB client interface.
Single Responsibility: Handle database connections and query execution.
"""
import asyncio
from typing import Any, Dict, List, Optional
from surrealdb import Surreal


class DatabaseConnectionError(Exception):
    """Raised when database connection fails."""
    pass


class QueryExecutionError(Exception):
    """Raised when query execution fails."""
    pass


class SurrealDBClient:
    """
    Database client for SurrealDB interactions.
    Follows Interface Segregation - provides only database operations.
    """

    def __init__(
        self,
        url: str = "ws://localhost:8000/rpc",
        namespace: str = "umls",
        database: str = "kg",
        username: Optional[str] = None,
        password: Optional[str] = None,
    ):
        """
        Initialize the SurrealDB client.
        
        Args:
            url: SurrealDB connection URL
            namespace: Database namespace
            database: Database name
            username: Optional username for authentication
            password: Optional password for authentication
        """
        self.url = url
        self.namespace = namespace
        self.database = database
        self.username = username
        self.password = password
        self.db: Optional[Surreal] = None
        self._connected = False

    def connect(self) -> None:
        """
        Establish connection to SurrealDB.
        
        Raises:
            DatabaseConnectionError: If connection fails
        """
        try:
            self.db = Surreal(self.url)
            
            # Sign in if credentials provided
            if self.username and self.password:
                self.db.signin({
                    "username": self.username,
                    "password": self.password
                })
            
            # Use namespace and database
            self.db.use(self.namespace, self.database)
            self._connected = True
        except Exception as e:
            raise DatabaseConnectionError(f"Connection failed: {e}") from e

    def close(self) -> None:
        """Close the database connection."""
        if self.db and self._connected:
            try:
                self.db.close()
                self._connected = False
            except Exception as e:
                # Log warning but don't raise
                pass

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

    async def create(
        self, 
        table: str, 
        data: Dict[str, Any], 
        record_id: Optional[str] = None
    ) -> Any:
        """
        Create a record in a table.
        
        Args:
            table: Table name
            data: Record data
            record_id: Optional record ID (e.g., "123")
            
        Returns:
            Created record
        """
        if not self._connected or not self.db:
            raise DatabaseConnectionError("Not connected to database")
        
        try:
            if record_id:
                full_id = f"{table}:{record_id}"
                return await asyncio.to_thread(self.db.create, full_id, data)
            return await asyncio.to_thread(self.db.create, table, data)
        except Exception as e:
            raise QueryExecutionError(f"Create failed: {e}") from e

    async def update(self, record_id: str, data: Dict[str, Any]) -> Any:
        """
        Update a record.
        
        Args:
            record_id: Full record ID (e.g., "concept:123")
            data: Update data
            
        Returns:
            Updated record
        """
        if not self._connected or not self.db:
            raise DatabaseConnectionError("Not connected to database")
        
        try:
            return await asyncio.to_thread(self.db.update, record_id, data)
        except Exception as e:
            raise QueryExecutionError(f"Update failed: {e}") from e

    async def upsert(
        self, 
        table: str, 
        data: Dict[str, Any], 
        record_id: str
    ) -> Any:
        """
        Upsert (insert or update) a record.
        
        Args:
            table: Table name
            data: Record data
            record_id: Record ID without table prefix (e.g., "123")
            
        Returns:
            Upserted record
        """
        if not self._connected or not self.db:
            raise DatabaseConnectionError("Not connected to database")
        
        full_id = f"{table}:{record_id}"
        
        # Try to update first, if it doesn't exist, create
        try:
            existing = await asyncio.to_thread(self.db.select, full_id)
            if existing:
                return await self.update(full_id, data)
        except:
            pass
        
        return await self.create(table, data, record_id)

    async def select(self, record_id: str) -> Optional[Any]:
        """
        Select a record by ID.
        
        Args:
            record_id: Full record ID (e.g., "concept:123")
            
        Returns:
            Record or None
        """
        if not self._connected or not self.db:
            raise DatabaseConnectionError("Not connected to database")
        
        try:
            return await asyncio.to_thread(self.db.select, record_id)
        except:
            return None

    async def delete(self, record_id: str) -> Any:
        """
        Delete a record.
        
        Args:
            record_id: Full record ID (e.g., "concept:123")
            
        Returns:
            Deleted record
        """
        if not self._connected or not self.db:
            raise DatabaseConnectionError("Not connected to database")
        
        try:
            return await asyncio.to_thread(self.db.delete, record_id)
        except Exception as e:
            raise QueryExecutionError(f"Delete failed: {e}") from e

    async def batch_upsert(
        self,
        table: str,
        records: List[Dict[str, Any]],
        id_field: str = "id",
    ) -> List[Any]:
        """
        Batch upsert multiple records.
        
        Args:
            table: Table name
            records: List of record dictionaries
            id_field: Field name containing the ID value
            
        Returns:
            List of upserted records
        """
        results = []
        for record in records:
            record_id = str(record[id_field])
            result = await self.upsert(table, record, record_id)
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


# Synchronous wrapper for convenience
class SurrealDBSync:
    """Synchronous wrapper for SurrealDB operations."""
    
    def __init__(self, *args, **kwargs):
        """Initialize with same parameters as SurrealDBClient."""
        self.client = SurrealDBClient(*args, **kwargs)
        self._loop = None
        self._connected = False
    
    def _get_loop(self):
        """Get or create event loop."""
        try:
            return asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop
    
    def connect(self):
        """Connect synchronously."""
        self.client.connect()
        self._connected = True
    
    def close(self):
        """Close synchronously."""
        self.client.close()
        self._connected = False
    
    def query(self, sql: str, vars: Optional[Dict[str, Any]] = None):
        """Query synchronously."""
        loop = self._get_loop()
        return loop.run_until_complete(self.client.execute_query(sql, vars))
    
    def create(self, table: str, data: Dict[str, Any], record_id: Optional[str] = None):
        """Create synchronously."""
        loop = self._get_loop()
        return loop.run_until_complete(self.client.create(table, data, record_id))
    
    def update(self, record_id: str, data: Dict[str, Any]):
        """Update synchronously."""
        loop = self._get_loop()
        return loop.run_until_complete(self.client.update(record_id, data))
    
    def upsert(self, table: str, data: Dict[str, Any], record_id: str):
        """Upsert synchronously."""
        loop = self._get_loop()
        return loop.run_until_complete(self.client.upsert(table, data, record_id))
    
    def select(self, record_id: str):
        """Select synchronously."""
        loop = self._get_loop()
        return loop.run_until_complete(self.client.select(record_id))
    
    def delete(self, record_id: str):
        """Delete synchronously."""
        loop = self._get_loop()
        return loop.run_until_complete(self.client.delete(record_id))
    
    def batch_upsert(self, table: str, records: List[Dict[str, Any]], id_field: str = "id"):
        """Batch upsert synchronously."""
        loop = self._get_loop()
        return loop.run_until_complete(self.client.batch_upsert(table, records, id_field))
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
