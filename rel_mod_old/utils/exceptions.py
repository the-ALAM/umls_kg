"""
Custom exception classes.
Single Responsibility: Define application-specific exceptions.
"""


class GraphTraversalException(Exception):
    """Base exception for graph traversal errors."""
    pass


class DatabaseConnectionError(GraphTraversalException):
    """Raised when database connection fails."""
    pass


class QueryExecutionError(GraphTraversalException):
    """Raised when query execution fails."""
    pass


class ConceptResolutionError(GraphTraversalException):
    """Raised when concept name to ID resolution fails."""
    pass


class FileProcessingError(GraphTraversalException):
    """Raised when file operations fail."""
    pass


class DataValidationError(GraphTraversalException):
    """Raised when data validation fails."""
    pass


class ConfigurationError(GraphTraversalException):
    """Raised when configuration is invalid."""
    pass