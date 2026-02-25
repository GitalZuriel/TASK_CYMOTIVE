"""
Embedding model abstraction supporting OpenAI and local sentence-transformers models.

Provides a unified interface for generating embeddings, with a factory function
to switch between providers based on configuration.
"""

import logging
from abc import ABC, abstractmethod

from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


class BaseEmbeddingModel(ABC):
    """Abstract interface for embedding models."""

    @abstractmethod
    def get_langchain_embeddings(self) -> Embeddings:
        """Return a LangChain-compatible Embeddings instance."""
        ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        ...

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model identifier."""
        ...


class OpenAIEmbeddingModel(BaseEmbeddingModel):
    """OpenAI text-embedding-3-small via LangChain."""

    def __init__(self) -> None:
        from langchain_openai import OpenAIEmbeddings
        self._model = OpenAIEmbeddings(model="text-embedding-3-small")
        logger.info("Initialized OpenAI embedding model: text-embedding-3-small")

    def get_langchain_embeddings(self) -> Embeddings:
        return self._model

    def embed_query(self, text: str) -> list[float]:
        return self._model.embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._model.embed_documents(texts)

    @property
    def model_name(self) -> str:
        return "openai/text-embedding-3-small"


class LocalEmbeddingModel(BaseEmbeddingModel):
    """Local all-MiniLM-L6-v2 via sentence-transformers, wrapped for LangChain."""

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer
        self._st_model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Initialized local embedding model: all-MiniLM-L6-v2")

    def get_langchain_embeddings(self) -> Embeddings:
        """Return a LangChain-compatible wrapper around sentence-transformers."""
        return _SentenceTransformerLangChainWrapper(self._st_model)

    def embed_query(self, text: str) -> list[float]:
        return self._st_model.encode(text, convert_to_numpy=True).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._st_model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    @property
    def model_name(self) -> str:
        return "sentence-transformers/all-MiniLM-L6-v2"


class _SentenceTransformerLangChainWrapper(Embeddings):
    """Thin LangChain Embeddings adapter for a SentenceTransformer model."""

    def __init__(self, model) -> None:
        self._model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode(text, convert_to_numpy=True).tolist()


def get_embedding_model(provider: str = "openai") -> BaseEmbeddingModel:
    """
    Factory function to create an embedding model instance.

    Args:
        provider: Either "openai" for text-embedding-3-small or
                  "local" for all-MiniLM-L6-v2.

    Returns:
        A BaseEmbeddingModel instance.

    Raises:
        ValueError: If provider is not recognized.
    """
    if provider == "openai":
        return OpenAIEmbeddingModel()
    elif provider == "local":
        return LocalEmbeddingModel()
    else:
        raise ValueError(f"Unknown embedding provider: '{provider}'. Use 'openai' or 'local'.")
