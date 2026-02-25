"""
ChromaDB vector store for automotive cybersecurity incidents.

Handles initialization, ingestion from incidents.json, and similarity queries.
Supports both OpenAI and local embedding models via the embeddings module.
"""

import json
import logging
from dataclasses import dataclass

import chromadb
from chromadb.config import Settings

import config
from src.embeddings import BaseEmbeddingModel

logger = logging.getLogger(__name__)


@dataclass
class VectorSearchResult:
    """A single result from a vector similarity search."""
    incident_id: str
    title: str
    description: str
    score: float  # distance — lower is more similar for ChromaDB L2/cosine
    metadata: dict


class VectorStore:
    """ChromaDB-backed vector store for incident embeddings."""

    def __init__(self, embedding_model: BaseEmbeddingModel, persist_dir: str | None = None) -> None:
        """
        Initialize the vector store.

        Args:
            embedding_model: The embedding model to use for encoding documents and queries.
            persist_dir: Directory for ChromaDB persistent storage. Defaults to config value.
        """
        self._embedding_model = embedding_model
        self._persist_dir = persist_dir or config.CHROMA_PERSIST_DIR

        self._client = chromadb.PersistentClient(
            path=self._persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=config.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB initialized (dir=%s, collection=%s, count=%d)",
            self._persist_dir,
            config.CHROMA_COLLECTION_NAME,
            self._collection.count(),
        )

    def ingest_incidents(self, incidents_path: str | None = None) -> int:
        """
        Load incidents from JSON and upsert into ChromaDB.

        Args:
            incidents_path: Path to incidents JSON file. Defaults to config value.

        Returns:
            Number of incidents ingested.
        """
        path = incidents_path or config.INCIDENTS_PATH
        with open(path, "r", encoding="utf-8") as f:
            incidents = json.load(f)

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []

        for inc in incidents:
            # Combine key fields for richer embedding (matches BM25 coverage)
            doc_text = (
                f"{inc['title']}. {inc['description']} "
                f"Attack vector: {inc['attack_vector']}. "
                f"Affected system: {inc['affected_system']}."
            )
            ids.append(inc["id"])
            documents.append(doc_text)
            metadatas.append({
                "title": inc["title"],
                "date": inc["date"],
                "severity": inc["severity"],
                "severity_score": inc.get("severity_score", 0),
                "attack_vector": inc["attack_vector"],
                "affected_system": inc["affected_system"],
                "cve": inc.get("cve") or "",
                "mitre_tactics": ",".join(inc.get("mitre_tactics", [])),
                "protocols": ",".join(
                    inc.get("indicators", {}).get("protocols", [])
                ),
                "components": ",".join(
                    inc.get("indicators", {}).get("components", [])
                ),
            })

        # Generate embeddings
        embeddings = self._embedding_model.embed_documents(documents)

        # Upsert into ChromaDB
        self._collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.info("Ingested %d incidents into ChromaDB", len(incidents))
        return len(incidents)

    def query(self, query_text: str, top_k: int | None = None) -> list[VectorSearchResult]:
        """
        Query the vector store for similar incidents.

        Args:
            query_text: The search query.
            top_k: Number of results to return. Defaults to config.TOP_K_RETRIEVAL.

        Returns:
            List of VectorSearchResult sorted by relevance (best first).
        """
        top_k = top_k or config.TOP_K_RETRIEVAL
        query_embedding = self._embedding_model.embed_query(query_text)

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        search_results: list[VectorSearchResult] = []
        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                search_results.append(VectorSearchResult(
                    incident_id=doc_id,
                    title=results["metadatas"][0][i].get("title", ""),
                    description=results["documents"][0][i],
                    score=results["distances"][0][i],
                    metadata=results["metadatas"][0][i],
                ))

        logger.info("Vector search returned %d results for query", len(search_results))
        return search_results

    @property
    def count(self) -> int:
        """Number of documents in the collection."""
        return self._collection.count()
