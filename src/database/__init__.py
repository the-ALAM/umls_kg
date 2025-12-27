"""Database operations and schema management for SurrealDB."""

from .surreal_client import SurrealDBClient, SurrealDBSync, DatabaseConnectionError, QueryExecutionError
from .queries import QueryBuilder
from .indexes import IndexManager, create_optimized_indexes
from .schema import load_schema_file, apply_schema

__all__ = [
    'SurrealDBClient',
    'SurrealDBSync',
    'DatabaseConnectionError',
    'QueryExecutionError',
    'QueryBuilder',
    'IndexManager',
    'create_optimized_indexes',
    'load_schema_file',
    'apply_schema',
]
