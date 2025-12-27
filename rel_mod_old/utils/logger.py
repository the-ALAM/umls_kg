"""
Logging utilities.
Single Responsibility: Centralized logging configuration and helpers.
"""
import logging
import sys
from typing import Optional
from pathlib import Path


class LoggerFactory:
    """Factory for creating configured loggers."""

    _loggers = {}

    @classmethod
    def get_logger(
        cls,
        name: str,
        level: int = logging.INFO,
        log_file: Optional[str] = None
    ) -> logging.Logger:
        """
        Get or create a logger with the specified configuration.
        
        Args:
            name: Logger name
            level: Logging level
            log_file: Optional file path for logging
            
        Returns:
            Configured logger instance
        """
        if name in cls._loggers:
            return cls._loggers[name]

        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.handlers.clear()

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # File handler if specified
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

        cls._loggers[name] = logger
        return logger


def log_execution(logger: logging.Logger):
    """
    Decorator to log function execution.
    
    Usage:
        @log_execution(logger)
        def my_function():
            pass
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.info(f"Executing {func.__name__}")
            try:
                result = func(*args, **kwargs)
                logger.info(f"Completed {func.__name__}")
                return result
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
                raise
        return wrapper
    return decorator
