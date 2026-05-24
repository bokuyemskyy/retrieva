from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

from slugify import slugify

from core.gpu_manager import GPUManager
from core.ingestion.chunker import Chunker
from core.ingestion.image_captioner import ImageCaptioner
from core.ingestion.processor.audio_processor import AudioProcessor
from core.ingestion.processor.base_file_processor import BaseFileProcessor
from core.ingestion.processor.document_processor import DocumentProcessor
from core.ingestion.processor.image_processor import ImageProcessor
from core.ingestion.processor.text_processor import TextProcessor
from embedding.embedder import BaseEmbedder
from models import Chunk, ChunkRecord, Document, DocumentRecord, SearchResult
from retrieval.retriever import DenseRetriever
from storage.object_storage import ObjectStorage
from storage.vector_storage import PostgresVectorStorage


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
        llm_model: str,
        system_prompt: Optional[str] = None,
    ):
        self.workspace = slugify(workspace)
        self.storage_dir = storage_dir
        self.embedder = embedder
        self.llm_client = llm_client
        self.chunker = chunker
        self.image_captioner = image_captioner
        self.llm_model = llm_model
        self.gpu_manager = GPUManager()

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
        ids = self.add_documents([original_path], metadata=metadata)
        return ids[0]

    def add_documents(
        self,
        original_paths: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[UUID]:
        if not original_paths:
            return []
        self.gpu_manager.activate(self.image_captioner.vlm)

        prepared: List[Tuple[Document, List[Chunk]]] = []
        document_ids: List[UUID] = []

        for path_str in original_paths:
            doc, chunks = self._prepare_document(path_str)
            prepared.append((doc, chunks))
            document_ids.append(doc.document_id)

        all_chunks: List[Chunk] = [c for _, chunks in prepared for c in chunks]
        self.gpu_manager.activate(self.embedder)
        self.embedder.embed_chunks(all_chunks)

        self.gpu_manager.deactivate()

        for doc, chunks in prepared:
            self._persist(doc, chunks)

        return document_ids

    def retrieve_chunks(
        self, query: str, top_k: int = 5, fetch_k: int = 30
    ) -> List[SearchResult]:
        with self.gpu_manager.using(self.embedder):
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
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()

    def get_chunks(self, limit: int = 10_000) -> List[ChunkRecord]:
        return self.vector_storage.get_chunks(workspace=self.workspace, limit=limit)

    def get_documents(self, limit: int = 10_000) -> List[DocumentRecord]:
        return self.vector_storage.get_documents(workspace=self.workspace, limit=limit)

    def delete_document(self, document_id: UUID) -> None:
        self.vector_storage.delete_document(document_id)
        self.object_storage.delete_file(
            workspace=self.workspace, document_id=document_id
        )

    def delete_all_workspaces(self) -> None:
        self.vector_storage.delete_all_workspaces()
        self.object_storage.delete_all_workspaces()

    def delete_current_workspace(self) -> None:
        self.vector_storage.delete_workspace(self.workspace)
        self.object_storage.delete_workspace(self.workspace)

    def _prepare_document(
        self,
        original_path: str,
    ) -> Tuple[Document, List[Chunk]]:
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
            chunks = processor.ingest(document)
            return document, chunks
        except Exception:
            try:
                self.object_storage.delete_file(self.workspace, document_id)
            except Exception:
                pass
            raise

    def _persist(self, document: Document, chunks: List[Chunk]) -> None:
        try:
            self.vector_storage.upsert_document(document)
            self.vector_storage.upsert_chunks(chunks)
        except Exception:
            try:
                self.vector_storage.delete_document(document.document_id)
            except Exception:
                pass
            try:
                self.object_storage.delete_file(self.workspace, document.document_id)
            except Exception:
                pass
            raise
