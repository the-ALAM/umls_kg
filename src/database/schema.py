"""Python schema management for SurrealDB."""

from pathlib import Path
from typing import Optional


def load_schema_file(schema_path: Optional[Path] = None) -> str:
    """
    Load SurrealDB schema SQL from file.
    
    Args:
        schema_path: Path to schema file. If None, uses default location.
        
    Returns:
        Schema SQL as string
    """
    if schema_path is None:
        schema_path = Path(__file__).parent.parent.parent / "config" / "surreal_schema.surql"
    
    return schema_path.read_text()


def apply_schema(client, schema_path: Optional[Path] = None) -> None:
    """
    Apply schema to SurrealDB instance.
    
    Args:
        client: SurrealDB client instance
        schema_path: Path to schema file. If None, uses default location.
    """
    schema_sql = load_schema_file(schema_path)
    
    # Execute schema statements
    for statement in schema_sql.split(";"):
        statement = statement.strip()
        if statement and not statement.startswith("--"):
            try:
                client.query(statement)
            except Exception as e:
                # Some statements might fail if schema already exists
                # This is acceptable for idempotent operations
                print(f"Warning: Schema statement may have failed: {e}")

