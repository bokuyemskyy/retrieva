from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from uuid import UUID, uuid4

from slugify import slugify

from core.embedding.embedder import BaseEmbedder, EmbedderConfig, EmbedderFactory
from core.ingestion.chunker import BaseChunker, RecursiveChunker
from core.ingestion.image_captioner import (
    BaseVLM,
    ImageCaptioner,
    VLMConfig,
    VLMFactory,
)
from core.ingestion.processor.audio_processor import AudioProcessor
from core.ingestion.processor.base_file_processor import BaseFileProcessor
from core.ingestion.processor.document_processor import DocumentProcessor
from core.ingestion.processor.image_processor import ImageProcessor
from core.ingestion.processor.text_processor import TextProcessor
from core.models import Chunk, Document, SearchResult
from core.storage.object_storage import ObjectStorage
from core.storage.vector_storage import Workspace, WorkspaceConfig, WorkspaceManager
from core.retrieval.llm import BaseLLM, LLMConfig, LLMFactory
from core.retrieval.reranker import BaseReranker, CrossEncoderReranker

DocumentInput = Union[str, Path, Tuple[str, bytes]]


class RAG:
    def __init__(
        self,
        storage_dir: str,
        postgres_url: str,
        system_prompt: Optional[str] = None,
    ):
        self.storage_dir = storage_dir

        self.chunker: BaseChunker = RecursiveChunker(
            chunk_size=800,
            chunk_overlap=120,
        )

        self.reranker: Optional[BaseReranker] = CrossEncoderReranker()

        self.llm_model: Optional[BaseLLM] = None
        self.vlm_model: Optional[BaseVLM] = None
        self.image_captioner: Optional[ImageCaptioner] = None
        self.processors: Dict[str, BaseFileProcessor] = {}

        self.object_storage = ObjectStorage(storage_dir)
        self.workspace_manager = WorkspaceManager(postgres_url)

        self.set_vlm(VLMConfig(provider="null", model_name="null"))

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
        assert self.active_workspace is not None, "Workspace is not active."
        return self.active_workspace

    @property
    def _embedder(self) -> BaseEmbedder:
        assert self.active_embedder is not None, "Embedder is not active."
        return self.active_embedder

    @property
    def _workspace_name(self) -> str:
        assert self.active_workspace_name is not None, "Workspace name is missing."
        return self.active_workspace_name

    def _ensure_workspace(self) -> None:
        if not (
            self.active_workspace
            and self.active_workspace_name
            and self.active_embedder
        ):
            raise RuntimeError(
                "No workspace selected. Please call 'select_workspace(name)'."
            )

    def set_llm(self, config: LLMConfig) -> None:
        self.llm_model = LLMFactory.create(config)

    def set_vlm(self, config: VLMConfig) -> None:
        self.vlm_model = VLMFactory.create(config)
        self.image_captioner = ImageCaptioner(vlm=self.vlm_model)
        self._refresh_processors()

    def set_system_prompt(self, prompt: str) -> None:
        self.system_prompt = prompt

    def _refresh_processors(self) -> None:
        if self.image_captioner is None:
            return

        doc_proc = DocumentProcessor(
            chunker=self.chunker, image_captioner=self.image_captioner
        )
        img_proc = ImageProcessor(
            chunker=self.chunker, image_captioner=self.image_captioner
        )
        txt_proc = TextProcessor(chunker=self.chunker)
        aud_proc = AudioProcessor(chunker=self.chunker)

        self.processors = {
            "pdf": doc_proc,
            "txt": txt_proc,
            "md": txt_proc,
            "png": img_proc,
            "jpg": img_proc,
            "jpeg": img_proc,
            "wav": aud_proc,
            "mp3": aud_proc,
        }

    def select_workspace(self, name: str) -> None:
        slug = slugify(name)
        ws = self.workspace_manager.workspace(slug)

        self.active_workspace = ws
        self.active_workspace_name = slug
        self.active_embedder = EmbedderFactory.create(config=ws.config.embedder_config)

    def create_workspace(self, name: str, embedder_config: EmbedderConfig) -> Workspace:
        slug = slugify(name)
        embedder = EmbedderFactory.create(embedder_config)

        ws = self.workspace_manager.create_workspace(
            name=slug, embedder_config=embedder_config, vector_size=embedder.vector_size
        )

        self.object_storage._workspace_dir(slug).mkdir(parents=True, exist_ok=True)
        return ws

    def get_workspaces(self) -> List[WorkspaceConfig]:
        return self.workspace_manager.get_workspaces()

    def delete_workspace(self, name: str) -> None:
        slug = slugify(name)
        self.workspace_manager.delete_workspace(slug)
        self.object_storage.delete_workspace(slug)

        if self.active_workspace_name == slug:
            self.active_workspace_name = None
            self.active_workspace = None
            self.active_embedder = None

    def delete_all_workspaces(self) -> None:
        self.workspace_manager.delete_all_workspaces()
        self.object_storage.delete_all_workspaces()
        self.active_workspace_name = None
        self.active_workspace = None
        self.active_embedder = None

    def add_document(self, document: DocumentInput) -> UUID:
        self._ensure_workspace()
        ids = self.add_documents([document])
        return ids[0]

    def add_documents(self, documents: List[DocumentInput]) -> List[UUID]:
        self._ensure_workspace()

        if not documents:
            return []

        prepared: List[Tuple[Document, List[Chunk]]] = []
        document_ids: List[UUID] = []
        chunks_to_embed: List[Chunk] = []

        for doc_input in documents:
            if isinstance(doc_input, tuple):
                filename, data = doc_input
                original_path = filename
                content_hash = hashlib.sha256(data).hexdigest()
                ext = Path(filename).suffix.lower().lstrip(".")

            else:
                path = Path(doc_input).expanduser().resolve()
                if not path.exists():
                    raise FileNotFoundError(f"Input file does not exist: {path}")

                filename = path.name
                original_path = str(path)
                data = None
                content_hash = self._calculate_hash(path)
                ext = path.suffix.lower().lstrip(".")

            existing_id = self._workspace.get_document_by_hash(content_hash)
            if existing_id:
                print(f"Skipping {filename}: Document already exists.")
                document_ids.append(existing_id)
                continue

            document_id = uuid4()
            document_ids.append(document_id)

            stored_path = self.object_storage.save_file(
                workspace=self._workspace_name,
                document_id=document_id,
                source_path=original_path,
                data=data,
            )

            document = Document(
                document_id=document_id,
                filename=filename,
                source_path=stored_path,
                original_path=original_path,
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
                doc_chunks = [
                    c for c in valid_chunks if c.document_id == doc.document_id
                ]
                doc_embeddings = [
                    v
                    for c, v in zip(valid_chunks, valid_embeddings)
                    if c.document_id == doc.document_id
                ]

                if doc_chunks:
                    self._persist(doc, doc_chunks, doc_embeddings)

        return document_ids

    def retrieve_chunks(self, query: str, top_k: int = 5) -> List[SearchResult]:
        self._ensure_workspace()
        query_vector = self._embedder.embed_query(query)

        fetch_k = top_k * 3 if self.reranker else top_k

        raw_results = self._workspace.hybrid_search(
            query_text=query,
            query_vector=query_vector,
            top_k=fetch_k,
            dense_weight=0.6,
            sparse_weight=0.4,
        )

        results = [
            SearchResult(
                chunk_id=res.chunk_id,
                content=res.content,
                score=res.score,
                document_id=uuid4(),
                metadata={},
            )
            for res in raw_results
        ]

        if self.reranker and results:
            new_scores = self.reranker.rerank(query, results)  # type: ignore
            for res, new_score in zip(results, new_scores):
                res.score = new_score
            results.sort(key=lambda x: x.score, reverse=True)
            results = results[:top_k]

        return results

    def generate_response(self, query: str, chunks: List[SearchResult]) -> str:
        if not chunks:
            return "No relevant information was found in the knowledge base."
        if not self.llm_model:
            return "No LLM model selected."

        context = "\n\n".join(f"[score={c.score:.3f}]\n{c.content}" for c in chunks)
        prompt = self.system_prompt.format(query=query, context=context)
        return self.llm_model.generate("", prompt)

    def generate_response_stream(self, query: str, chunks: List[SearchResult]):
        if not chunks:
            yield "No relevant information was found in the knowledge base."
            return
        if not self.llm_model:
            yield "No LLM model selected."
            return

        context = "\n\n".join(f"[score={c.score:.3f}]\n{c.content}" for c in chunks)
        prompt = self.system_prompt.format(query=query, context=context)
        yield from self.llm_model.generate_stream("", prompt)

    def query(self, query: str, top_k: int = 5) -> str:
        chunks = self.retrieve_chunks(query, top_k=top_k)
        return self.generate_response(query, chunks)

    def query_stream(self, query: str, top_k: int = 5):
        chunks = self.retrieve_chunks(query, top_k=top_k)
        yield from self.generate_response_stream(query, chunks)

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
            while chunk := f.read(8192):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
