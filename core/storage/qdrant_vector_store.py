from __future__ import annotations

from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointIdsList,
    PointStruct,
    VectorParams,
)

from core.model.chunk import Chunk


class QdrantVectorStore:
    def __init__(
        self,
        url: str,
        workspace: str,
        vector_size: int = 1536,
    ):
        self.client = QdrantClient(url=url)
        self.workspace = workspace
        self.vector_size = vector_size

        self.documents_collection = f"rag_{workspace}_documents"
        self.chunks_collection = f"rag_{workspace}_chunks"

        self._init_collections()

    def _init_collections(self) -> None:
        for name in [self.documents_collection, self.chunks_collection]:
            if not self.client.collection_exists(name):
                self.client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE,
                    ),
                )

    def upsert_document(self, doc: Dict[str, Any]) -> None:
        self.client.upsert(
            collection_name=self.documents_collection,
            points=[
                PointStruct(
                    id=doc["document_id"],
                    vector=[0.0] * self.vector_size,
                    payload=doc,
                )
            ],
        )

    def upsert_chunks(self, chunks: List[Chunk]) -> None:
        points = []
        for c in chunks:
            if c.embedding is None:
                raise ValueError(
                    f"Chunk {c.chunk_id} has no embedding. "
                    "Call embedder.embed_chunks() before upsert_chunks()."
                )
            points.append(
                PointStruct(
                    id=c.chunk_id,
                    vector=c.embedding,
                    payload={
                        "document_id": c.document_id,
                        "text": c.content,
                        "source_path": c.source_path,
                        "modality": c.modality.value,
                        **(c.metadata or {}),
                    },
                )
            )

        if points:
            self.client.upsert(
                collection_name=self.chunks_collection,
                points=points,
            )

    def search(
        self,
        query_vector: List[float],
        top_k: int,
        collection: Optional[str] = None,
    ):
        response = self.client.query_points(
            collection_name=collection or self.chunks_collection,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )
        return response.points

    def get_all_chunks(self, limit: int = 10_000) -> List[Dict[str, Any]]:
        results, _ = self.client.scroll(
            collection_name=self.chunks_collection,
            limit=limit,
            with_payload=True,
            with_vector=False,
        )
        return [
            {
                "chunk_id": str(point.id),
                "document_id": point.payload.get("document_id"),
                "text": point.payload.get("text"),
                "metadata": point.payload,
            }
            for point in results
        ]

    def get_all_documents(self, limit: int = 10_000) -> List[Dict[str, Any]]:
        results, _ = self.client.scroll(
            collection_name=self.documents_collection,
            limit=limit,
            with_payload=True,
            with_vector=False,
        )
        return [point.payload for point in results]

    def delete_document(self, document_id: str) -> None:
        self.client.delete(
            collection_name=self.chunks_collection,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id),
                    )
                ]
            ),
        )
        self.client.delete(
            collection_name=self.documents_collection,
            points_selector=PointIdsList(points=[document_id]),
        )

    def delete_all_chunks_and_documents(self) -> None:
        self.client.delete(
            collection_name=self.chunks_collection, points_selector=Filter()
        )
        self.client.delete(
            collection_name=self.documents_collection, points_selector=Filter()
        )

    def delete_workspace_collections(self) -> None:
        if self.client.collection_exists(self.documents_collection):
            self.client.delete_collection(self.documents_collection)
        if self.client.collection_exists(self.chunks_collection):
            self.client.delete_collection(self.chunks_collection)
