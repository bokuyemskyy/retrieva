from __future__ import annotations

from typing import Optional, Dict, Any

from core.ingestion.chunker import Chunker
from core.embedding.base_embedder import BaseEmbedder
from core.ingestion.processor.base_file_processor import BaseFileProcessor
from core.storage.qdrant_vector_store import QdrantVectorStore
from core.storage.storage_manager import StorageManager


class Ingestor:
    def __init__(
        self,
        storage_manager: StorageManager,
        embedder: BaseEmbedder,
        vector_store: QdrantVectorStore,
        chunker: Optional[Chunker] = None,
        processors: Optional[Dict[str, BaseFileProcessor]] = None,
    ):
        self.storage = storage_manager
        self.embedder = embedder
        self.vector_store = vector_store
        self.chunker = chunker

        self.processors = processors or {}

    def save_and_process_file(
        self,
        source_path: str,
        file_type: str,
        metadata: Dict[str, Any],
    ) -> str:
        file_type = file_type.lower().lstrip(".")

        processor = self.processors.get(file_type)
        if not processor:
            raise ValueError(f"Unsupported file type: '{file_type}'. ")

        document_id, stored_path = self.storage.save_file(
            source_path=source_path,
            original_filename=metadata.get("filename", "unknown"),
        )

        chunks = processor.ingest(stored_path)

        if not chunks:
            self.vector_store.upsert_document(
                {
                    "document_id": document_id,
                    "filename": metadata.get("filename"),
                    "file_path": stored_path,
                    "type": file_type,
                    "chunk_count": 0,
                }
            )
            return document_id

        for chunk in chunks:
            chunk.document_id = document_id

        chunks = self.embedder.embed_chunks(chunks)

        self.vector_store.upsert_document(
            {
                "document_id": document_id,
                "filename": metadata.get("filename"),
                "file_path": stored_path,
                "type": file_type,
                "chunk_count": len(chunks),
            }
        )

        self.vector_store.upsert_chunks(chunks)

        return document_id
