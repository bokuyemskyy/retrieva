from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from uuid import UUID, uuid4

from core.embedding.embedder import BaseEmbedder, EmbedderConfig, EmbedderFactory
from slugify import slugify
from core.storage.vector_storage import VectorStorage, Workspace, WorkspaceConfig
from core.ingestion.chunker import Chunker
from core.ingestion.image_captioner import ImageCaptioner
from core.ingestion.processor.audio_processor import AudioProcessor
from core.ingestion.processor.base_file_processor import BaseFileProcessor
from core.ingestion.processor.document_processor import DocumentProcessor
from core.ingestion.processor.image_processor import ImageProcessor
from core.ingestion.processor.text_processor import TextProcessor
from core.models import Chunk, Document, SearchResult
from core.storage.object_storage import ObjectStorage


class RAG:
    def __init__(
        self,
        storage_dir: str,
        llm_client,
        llm_model: str,
        chunker: Chunker,
        image_captioner: ImageCaptioner,
        postgres_url: str,
        system_prompt: Optional[str] = None,
    ):
        self.storage_dir = storage_dir
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.chunker = chunker
        self.image_captioner = image_captioner

        self.object_storage = ObjectStorage(storage_dir)
        self.vector_storage = VectorStorage(postgres_url)

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

        self.active_workspace_name: Optional[str] = None
        self.active_workspace: Optional[Workspace] = None
        self.active_embedder: Optional[BaseEmbedder] = None

    @property
    def _workspace(self) -> Workspace:
        assert self.active_workspace is not None
        return self.active_workspace

    @property
    def _embedder(self) -> BaseEmbedder:
        assert self.active_embedder is not None
        return self.active_embedder

    @property
    def _workspace_name(self) -> str:
        assert self.active_workspace_name is not None
        return self.active_workspace_name

    def _ensure_workspace(self) -> None:
        if (
            self.active_workspace is None
            or self.active_workspace_name is None
            or self.active_embedder is None
        ):
            raise RuntimeError(
                "No workspace selected. Please call 'select_workspace(name)'."
            )

        assert self.active_workspace is not None

    def select_workspace(self, name: str) -> None:
        slug = slugify(name)

        # 1. Fetch the workspace wrapper from storage
        ws = self.vector_storage.workspace(slug)

        self.active_workspace = ws
        self.active_workspace_name = slug

        embedder_config = EmbedderConfig(
            provider=ws.config.embedding_provider,
            model_name=ws.config.embedding_model_name,
            api_key=ws.config.embedding_api_key,
            base_url=ws.config.embedding_base_url,
        )

        self.active_embedder = EmbedderFactory.create(config=embedder_config)

    def create_workspace(
        self, name: str, embedder: BaseEmbedder, config: Optional[dict] = None
    ) -> Workspace:
        slug = slugify(name)

        ws = self.vector_storage.create_workspace(
            name=slug, embedder=embedder, config=config
        )

        self.object_storage._workspace_dir(slug).mkdir(parents=True, exist_ok=True)
        return ws

    def get_workspaces(self) -> List[WorkspaceConfig]:
        return self.vector_storage.get_workspaces()

    def delete_workspace(self, name: str) -> None:
        slug = slugify(name)
        self.vector_storage.delete_workspace(slug)
        self.object_storage.delete_workspace(slug)

        if self.active_workspace_name == slug:
            self.active_workspace_name = None
            self.active_workspace = None
            self.active_embedder = None

    def delete_all_workspaces(self) -> None:
        self.vector_storage.delete_all_workspaces()
        self.object_storage.delete_all_workspaces()
        self.active_workspace_name = None
        self.active_workspace = None
        self.active_embedder = None

    def add_document(self, original_path: str) -> UUID:
        self._ensure_workspace()
        ids = self.add_documents([original_path])
        return ids[0]

    def add_documents(self, original_paths: List[str]) -> List[UUID]:
        self._ensure_workspace()

        if not original_paths:
            return []

        prepared: List[Tuple[Document, List[Chunk]]] = []
        document_ids: List[UUID] = []
        chunks_to_embed: List[Chunk] = []

        for path_str in original_paths:
            path = Path(path_str).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(f"Input file does not exist: {path}")

            content_hash = self._calculate_hash(path)

            existing_id = self._workspace.get_document_by_hash(content_hash)
            if existing_id:
                print(f"Skipping {path.name}: Document already exists.")
                document_ids.append(existing_id)
                continue

            document_id = uuid4()
            document_ids.append(document_id)

            stored_path = self.object_storage.save_file(
                workspace=self._workspace_name,
                document_id=document_id,
                source_path=str(path),
            )

            document = Document(
                document_id=document_id,
                filename=path.name,
                source_path=stored_path,
                original_path=str(path),
                content_hash=content_hash,
            )

            cached_chunks = self.object_storage.load_chunks_cache(
                self._workspace_name, content_hash
            )

            if cached_chunks is not None:
                for chunk in cached_chunks:
                    chunk.document_id = document_id
                chunks = cached_chunks
            else:
                ext = path.suffix.lower().lstrip(".")
                processor = self.processors.get(ext)
                if processor is None:
                    raise ValueError(f"Unsupported file type: {ext!r}")

                try:
                    chunks = processor.ingest(document)
                    self.object_storage.save_chunks_cache(
                        self._workspace_name, content_hash, chunks
                    )
                except Exception:
                    self.object_storage.delete_file(
                        self._workspace_name, document_id, ext
                    )
                    raise

            prepared.append((document, chunks))
            chunks_to_embed.extend(chunks)

        if chunks_to_embed:
            embeddings = self._embedder.embed_chunks(chunks_to_embed)

            valid_chunks = []
            valid_embeddings = []

            for chunk, vec in zip(chunks_to_embed, embeddings):
                if vec is not None:
                    valid_chunks.append(chunk)
                    valid_embeddings.append(vec)

            for doc, _ in prepared:
                doc_chunks = []
                doc_embeddings = []

                for c, v in zip(valid_chunks, valid_embeddings):
                    if c.document_id == doc.document_id:
                        doc_chunks.append(c)
                        doc_embeddings.append(v)

                self._persist(doc, doc_chunks, doc_embeddings)

        return document_ids

    def retrieve_chunks(self, query: str, top_k: int = 5) -> List[SearchResult]:
        self._ensure_workspace()

        query_vector = self._embedder.embed_query(query)

        raw_results = self._workspace.search(query_vector, top_k=top_k)

        return [
            SearchResult(
                chunk_id=res["chunk_id"],
                content=res["content"],
                score=res["score"],
                document_id=uuid4(),
                metadata={},
            )
            for res in raw_results
        ]

    def query(self, query: str, top_k: int = 5) -> str:
        chunks = self.retrieve_chunks(query, top_k=top_k)
        return self.generate_response(query, chunks)

    def generate_response(self, query: str, chunks: List[SearchResult]) -> str:
        if not chunks:
            return "No relevant information was found in the knowledge base."

        context = "\n\n".join(f"[score={c.score:.3f}]\n{c.content}" for c in chunks)
        prompt = self.system_prompt.format(query=query, context=context)

        response = self.llm_client.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()

    def get_chunks(self, limit: int = 10_000) -> List[Chunk]:
        self._ensure_workspace()

        return self._workspace.get_chunks(limit=limit)

    def get_documents(self, limit: int = 10_000) -> List[Document]:
        self._ensure_workspace()

        return self._workspace.get_documents(limit=limit)

    def delete_document(self, document_id: UUID) -> None:
        self._ensure_workspace()

        doc = self._workspace.get_document(document_id)
        if not doc:
            return

        ext = Path(doc.filename).suffix.lower().lstrip(".")

        self._workspace.delete_document(document_id)

        self.object_storage.delete_file(
            workspace=self._workspace_name, document_id=document_id, extension=ext
        )

    def _persist(
        self, document: Document, chunks: List[Chunk], embeddings: List[List[float]]
    ) -> None:
        self._ensure_workspace()

        try:
            self._workspace.upsert_document(document)
            self._workspace.upsert_chunks(chunks, embeddings)
        except Exception:
            try:
                self._workspace.delete_document(document.document_id)
            except Exception:
                pass

            try:
                ext = Path(document.filename).suffix.lower().lstrip(".")
                self.object_storage.delete_file(
                    self._workspace_name, document.document_id, ext
                )
            except Exception:
                pass
            raise

    def _calculate_hash(self, filepath: Path) -> str:
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
