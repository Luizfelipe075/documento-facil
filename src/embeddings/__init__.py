"""Embeddings package."""
from src.embeddings.embedder import Embedder
from src.embeddings.offline import HashingEmbedder

__all__ = ["Embedder", "HashingEmbedder"]
