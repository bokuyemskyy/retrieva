from typing import Optional

from core.ingestion.chunker import Chunker
from core.storage.qdrant_vector_store import QdrantVectorStore
from core.embedding.base_embedder import BaseEmbedder


class Ingestor:
    def __init__(
        self,
        storage_dir: str,
        embedder: BaseEmbedder,
        vector_store: QdrantVectorStore,
        chunker: Optional[Chunker] = None,
    ):
        self.storage_dir = storage_dir
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.processors = {"pdf": DocumentProcessor(), "md": TextProcessor()}

    def save_and_process_file(
        self, source_path: str, file_type: str, metadata: Dict[str, Any]
    ) -> str:
        file_id = "unique_file_uuid"
        internal_path = f"{self.storage_dir}/{file_id}.{file_type}"

        processor = self.processors.get(file_type.lower())
        if not processor:
            raise ValueError(f"Unsupported file type: {file_type}")
        raw_text = processor.extract_text(internal_path)

        chunks = self.chunker.split_text(raw_text, file_id, base_metadata=metadata)
        chunks = self.embedder.embed_chunks(chunks)

        self.vector_store.upsert_chunks(chunks)
        return file_id
