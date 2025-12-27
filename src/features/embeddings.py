"""Batch embedding generation pipeline using sentence-transformers."""

from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import torch


class EmbeddingGenerator:
    """Batch embedding generator for concept text."""
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        device: Optional[str] = None,
        batch_size: int = 10000,
    ):
        """
        Initialize embedding generator.
        
        Args:
            model_name: HuggingFace model name
            device: Device to use ('cuda', 'cpu', or None for auto)
            batch_size: Batch size for processing
        """
        self.model_name = model_name
        self.batch_size = batch_size
        
        # Auto-detect device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        # Load model
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
    
    def generate_embeddings(
        self,
        texts: List[str],
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            show_progress: Whether to show progress bar
            
        Returns:
            NumPy array of embeddings (n_samples, embedding_dim)
        """
        # Clean texts (remove empty strings)
        texts = [text if text else " " for text in texts]
        
        # Generate embeddings in batches
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,  # Normalize for cosine similarity
        )
        
        return embeddings
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text string
            
        Returns:
            Embedding vector
        """
        if not text:
            text = " "
        
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        
        return embedding


def create_embedding_generator(
    model_name: str = "BAAI/bge-large-en-v1.5",
    batch_size: int = 10000,
) -> EmbeddingGenerator:
    """
    Factory function to create an embedding generator.
    
    Args:
        model_name: HuggingFace model name
        batch_size: Batch size for processing
        
    Returns:
        EmbeddingGenerator instance
    """
    return EmbeddingGenerator(model_name=model_name, batch_size=batch_size)

