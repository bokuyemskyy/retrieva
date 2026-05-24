from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from ingestion.processor.base_file_processor import BaseFileProcessor
from models import ChunkRecord, Document, DocumentRecord, SearchResult
from slugify import slugify

from embedding.embedder import BaseEmbedder
from storage.vector_storage import PostgresVectorStorage
from core.ingestion.chunker import Chunker
from core.ingestion.processor.audio_processor import AudioProcessor
from core.ingestion.processor.document_processor import DocumentProcessor
from core.ingestion.processor.image_processor import ImageProcessor
from core.ingestion.processor.text_processor import TextProcessor
from retrieval.retriever import BaseRetriever, DenseRetriever
from storage.object_storage import ObjectStorage
from core.ingestion.image_captioner import ImageCaptioner


class RAG:
    def __init__(
        self,
        workspace: str,
        storage_dir: str,
        embedder: BaseEmbedder,
        llm_client,
        chunker: Chunker,
        image_captioner: ImageCaptioner,
        postgres_url: str,
        system_prompt: Optional[str] = None,
    ):
        self.workspace = slugify(workspace)
        self.storage_dir = storage_dir
        self.embedder = embedder
        self.llm_client = llm_client
        self.chunker = chunker
        self.image_captioner = image_captioner

        self.object_storage = ObjectStorage(storage_dir)
        self.vector_storage: PostgresVectorStorage = PostgresVectorStorage(postgres_url)

        self.retriever = DenseRetriever(self.vector_storage, self.embedder)

        self.processors: Dict[str, BaseFileProcessor] = {
            "pdf": DocumentProcessor(chunker=chunker, image_captioner=image_captioner),
            "txt": TextProcessor(chunker=chunker),
            "md": TextProcessor(chunker=chunker),
            "png": ImageProcessor(chunker=chunker, image_captioner=image_captioner),
            "jpg": ImageProcessor(chunker=chunker, image_captioner=image_captioner),
            "jpeg": ImageProcessor(chunker=chunker, image_captioner=image_captioner),
            "wav": AudioProcessor(chunker=chunker),
            "mp3": AudioProcessor(chunker=chunker),
        }

        self.system_prompt = system_prompt or (
            "Answer the question using only the provided context. "
            "If the context does not contain enough information, say so.\n\n"
            "Context:\n{context}\n\n"
            "Question: {query}"
        )

    def add_document(
        self,
        original_path: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> UUID:
        path = Path(original_path).expanduser().resolve()

        if not path.exists():
            raise FileNotFoundError(f"Input file does not exist: {path}")

        ext = path.suffix.lower().lstrip(".")

        processor = self.processors.get(ext)
        if processor is None:
            raise ValueError(f"Unsupported file type: {ext!r}")

        document_id = uuid4()
        stored_path = self.object_storage.save_file(
            workspace=self.workspace,
            document_id=document_id,
            source_path=str(path),
        )

        document = Document(
            document_id=document_id,
            workspace=self.workspace,
            filename=path.name,
            source_path=stored_path,
            original_path=str(path),
        )

        try:
            self.vector_storage.upsert_document(document)

            print("upsert complete")
            chunks = processor.ingest(document)

            print(f"ingestion complete, got {chunks}")
            self.embedder.embed_chunks(chunks)

            print("embedding complete")
            self.vector_storage.upsert_chunks(chunks)

            print("upsert complete")
            return document_id

        except Exception:
            try:
                self.vector_storage.delete_document(document_id)
            except Exception:
                pass

            try:
                self.object_storage.delete_file(self.workspace, document_id)
            except Exception:
                pass

            raise

    def retrieve_chunks(
        self, query: str, top_k: int = 5, fetch_k: int = 30
    ) -> List[SearchResult]:
        return self.retriever.retrieve(
            workspace=self.workspace, query=query, top_k=top_k
        )

    def generate_response(self, query: str, top_k: int = 5, fetch_k: int = 30) -> str:
        chunks = self.retrieve_chunks(query, top_k=top_k, fetch_k=fetch_k)

        if not chunks:
            return "No relevant information was found in the knowledge base."

        context = "\n\n".join(f"[score={c.score:.3f}]\n{c.content}" for c in chunks)

        prompt = self.system_prompt.format(query=query, context=context)

        response = self.llm_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
        )

        return response.choices[0].message.content.strip()

    def get_chunks(self, limit: int = 10_000) -> List[ChunkRecord]:
        return self.vector_storage.get_chunks(
            workspace=self.workspace,
            limit=limit,
        )

    def get_documents(self, limit: int = 10_000) -> List[DocumentRecord]:
        return self.vector_storage.get_documents(
            workspace=self.workspace,
            limit=limit,
        )

    def delete_document(self, document_id: UUID) -> None:
        self.vector_storage.delete_document(document_id)
        self.object_storage.delete_file(
            workspace=self.workspace,
            document_id=document_id,
        )

    def delete_all_workspaces(self) -> None:
        self.vector_storage.delete_all_workspaces()
        self.object_storage.delete_all_workspaces()

    def delete_current_workspace(self) -> None:
        self.vector_storage.delete_workspace(self.workspace)
        self.object_storage.delete_workspace(self.workspace)
