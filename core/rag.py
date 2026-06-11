from __future__ import annotations

import hashlib
from pathlib import Path
from uuid import UUID, uuid4

from slugify import slugify

from core.embedding.embedder import EmbedderConfig, EmbedderFactory
from core.ingestion.chunker import BaseChunker, RecursiveChunker
from core.ingestion.image_captioner import (
    ImageCaptioner,
    VLMConfig,
    VLMFactory,
)
from core.ingestion.processor.audio_processor import AudioProcessor
from core.ingestion.processor.document_processor import DocumentProcessor
from core.ingestion.processor.image_processor import ImageProcessor
from core.ingestion.processor.text_processor import TextProcessor
from core.models import (
    Chunk,
    ChunkSearchResult,
    ChunkWithEmbedding,
    Document,
    DocumentSearchResult,
    Embedding,
)
from core.storage.object_storage import ObjectStorage
from core.storage.vector_storage import Workspace, WorkspaceConfig, WorkspaceManager
from core.retrieval.llm import LLMConfig, LLMFactory
from core.retrieval.reranker import BaseReranker

DocumentInput = str | Path | tuple[str, bytes]


class RAG:
    def __init__(
        self,
        storage_dir: str,
        postgres_url: str,
    ):
        self.storage_dir = storage_dir
        self.object_storage = ObjectStorage(storage_dir)
        self.workspace_manager = WorkspaceManager(postgres_url)

        self.chunker: BaseChunker = RecursiveChunker(
            chunk_size=800,
            chunk_overlap=120,
        )

    def _get_processors(self, vlm_config: VLMConfig | None = None) -> dict:
        vlm = VLMFactory.create(
            vlm_config or VLMConfig(provider="null", model_name="null"), validate=False
        )
        captioner = ImageCaptioner(vlm=vlm)

        doc_proc = DocumentProcessor(chunker=self.chunker, image_captioner=captioner)
        img_proc = ImageProcessor(chunker=self.chunker, image_captioner=captioner)
        txt_proc = TextProcessor(chunker=self.chunker)
        aud_proc = AudioProcessor(chunker=self.chunker)

        return {
            "pdf": doc_proc,
            "txt": txt_proc,
            "md": txt_proc,
            "png": img_proc,
            "jpg": img_proc,
            "jpeg": img_proc,
            "wav": aud_proc,
            "mp3": aud_proc,
        }

    def create_workspace(self, name: str, embedder_config: EmbedderConfig) -> Workspace:
        slug = slugify(name)
        embedder = EmbedderFactory.create(embedder_config)

        ws = self.workspace_manager.create_workspace(
            name=slug, embedder_config=embedder_config, vector_size=embedder.vector_size
        )

        self.object_storage._workspace_dir(slug).mkdir(parents=True, exist_ok=True)
        return ws

    def get_workspaces(self) -> list[WorkspaceConfig]:
        return self.workspace_manager.get_workspaces()

    def delete_workspace(self, name: str) -> None:
        slug = slugify(name)
        self.workspace_manager.delete_workspace(slug)
        self.object_storage.delete_workspace(slug)

    def delete_all_workspaces(self) -> None:
        self.workspace_manager.delete_all_workspaces()
        self.object_storage.delete_all_workspaces()

    def add_document(
        self,
        workspace_name: str,
        document: DocumentInput,
        vlm_config: VLMConfig | None = None,
    ) -> UUID:
        ids = self.add_documents(workspace_name, [document], vlm_config)
        return ids[0]

    def add_documents(
        self,
        workspace_name: str,
        documents: list[DocumentInput],
        vlm_config: VLMConfig | None = None,
    ) -> list[UUID]:
        slug = slugify(workspace_name)
        workspace = self.workspace_manager.workspace(slug)
        embedder = EmbedderFactory.create(config=workspace.config.embedder_config)
        processors = self._get_processors(vlm_config)

        if not documents:
            return []

        prepared: list[tuple[Document, list[Chunk]]] = []
        document_ids: list[UUID] = []
        chunks_to_embed: list[Chunk] = []

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

            existing_id = workspace.get_document_by_hash(content_hash)
            if existing_id:
                document_ids.append(existing_id)
                continue

            document_id = uuid4()
            document_ids.append(document_id)

            stored_path = self.object_storage.save_file(
                workspace=slug,
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

            cached_chunks = self.object_storage.load_chunks_cache(slug, content_hash)

            if cached_chunks is not None:
                for chunk in cached_chunks:
                    chunk.document_id = document_id
                chunks = cached_chunks

            else:
                ext = Path(filename).suffix.lower().lstrip(".")
                processor = processors.get(ext)

                if processor is None:
                    raise ValueError(f"Unsupported file type: {ext!r}")

                chunks = processor.ingest(document)
                self.object_storage.save_chunks_cache(slug, content_hash, chunks)

            prepared.append((document, chunks))
            chunks_to_embed.extend(chunks)

        if chunks_to_embed:
            embeddings = embedder.embed_chunks(chunks_to_embed)

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
                    self._persist(workspace, slug, doc, doc_chunks, doc_embeddings)

        return document_ids

    def retrieve_chunks(
        self,
        workspace_name: str,
        query: str,
        top_k: int = 5,
        reranker: BaseReranker | None = None,
    ) -> list[ChunkSearchResult]:
        slug = slugify(workspace_name)
        workspace = self.workspace_manager.workspace(slug)
        embedder = EmbedderFactory.create(config=workspace.config.embedder_config)

        query_vector = embedder.embed_query(query)
        fetch_k = top_k * 3 if reranker else top_k

        results = workspace.hybrid_search(
            query_text=query,
            query_vector=query_vector,
            top_k=fetch_k,
            dense_weight=0.6,
            sparse_weight=0.4,
        )

        if reranker and results:
            chunks = [res.chunk for res in results]
            new_scores = reranker.rerank(query, chunks)
            for res, new_score in zip(results, new_scores):
                res.score = new_score
            results.sort(key=lambda x: x.score, reverse=True)
            results = results[:top_k]

        return results

    def search_files(
        self, workspace_name: str, query: str, top_k: int = 5
    ) -> list[DocumentSearchResult]:
        slug = slugify(workspace_name)
        workspace = self.workspace_manager.workspace(slug)

        chunk_results = self.retrieve_chunks(
            workspace_name, query, top_k=max(top_k * 5, 20)
        )

        doc_scores: dict[UUID, float] = {}
        for res in chunk_results:
            doc_id = res.chunk.document_id
            doc_scores[doc_id] = max(doc_scores.get(doc_id, 0.0), res.score)

        sorted_doc_ids = sorted(
            doc_scores.keys(), key=lambda d: doc_scores[d], reverse=True
        )[:top_k]

        top_files = []
        for doc_id in sorted_doc_ids:
            doc = workspace.get_document(doc_id)
            if doc:
                top_files.append(
                    DocumentSearchResult(document=doc, score=doc_scores[doc_id])
                )

        return top_files

    def generate_response(
        self,
        chunks: list[ChunkSearchResult],
        query: str,
        llm_config: LLMConfig,
        system_prompt: str | None = None,
    ) -> str:
        if not chunks:
            return "No relevant information was found in the knowledge base."

        llm_model = LLMFactory.create(llm_config, validate=False)

        default_prompt = (
            "You are an expert, truthful AI assistant. Your task is to answer the user's "
            "question explicitly and solely based on the provided context.\n\n"
            "<context>\n{context}\n</context>\n\n"
            "Instructions:\n"
            "1. STRICT ADHERENCE: Answer strictly based on the <context> above. Under no circumstances should you use outside knowledge or hallucinate information.\n"
            "2. MISSING INFO: If the exact answer cannot be deduced from the context, respond exactly with: 'I cannot answer this based on the provided context.' Do not attempt to guess.\n"
            "3. DIRECTNESS: Be direct, concise, and highly relevant to the question. Do not start your answer with filler like 'Based on the provided context...'.\n\n"
            "Question: {query}\n"
            "Answer:"
        )
        prompt_template = system_prompt or default_prompt

        context_blocks = [
            f"--- Document Snippet (Score: {c.score:.3f}) ---\n{c.chunk.content}"
            for c in chunks
        ]
        context = "\n\n".join(context_blocks)

        prompt = prompt_template.format(query=query, context=context)
        return llm_model.generate("", prompt)

    def generate_response_stream(
        self,
        chunks: list[ChunkSearchResult],
        query: str,
        llm_config: LLMConfig,
        system_prompt: str | None = None,
    ):
        if not chunks:
            yield "No relevant information was found in the knowledge base."
            return

        llm_model = LLMFactory.create(llm_config, validate=False)

        default_prompt = (
            "You are an expert, truthful AI assistant. Your task is to answer the user's "
            "question explicitly and solely based on the provided context.\n\n"
            "<context>\n{context}\n</context>\n\n"
            "Instructions:\n"
            "1. STRICT ADHERENCE: Answer strictly based on the <context> above. Under no circumstances should you use outside knowledge or hallucinate information.\n"
            "2. MISSING INFO: If the exact answer cannot be deduced from the context, respond exactly with: 'I cannot answer this based on the provided context.' Do not attempt to guess.\n"
            "3. DIRECTNESS: Be direct, concise, and highly relevant to the question. Do not start your answer with filler like 'Based on the provided context...'.\n\n"
            "Question: {query}\n"
            "Answer:"
        )
        prompt_template = system_prompt or default_prompt

        context_blocks = [
            f"--- Document Snippet (Score: {c.score:.3f}) ---\n{c.chunk.content}"
            for c in chunks
        ]
        context = "\n\n".join(context_blocks)

        prompt = prompt_template.format(query=query, context=context)
        yield from llm_model.generate_stream("", prompt)

    def query(
        self,
        workspace_name: str,
        query: str,
        llm_config: LLMConfig,
        top_k: int = 5,
        reranker: BaseReranker | None = None,
        system_prompt: str | None = None,
    ) -> str:
        chunks = self.retrieve_chunks(
            workspace_name, query, top_k=top_k, reranker=reranker
        )
        return self.generate_response(chunks, query, llm_config, system_prompt)

    def query_stream(
        self,
        workspace_name: str,
        query: str,
        llm_config: LLMConfig,
        top_k: int = 5,
        reranker: BaseReranker | None = None,
        system_prompt: str | None = None,
    ):
        chunks = self.retrieve_chunks(
            workspace_name, query, top_k=top_k, reranker=reranker
        )
        yield from self.generate_response_stream(
            chunks, query, llm_config, system_prompt
        )

    def get_chunks(self, workspace_name: str, limit: int = 10_000) -> list[Chunk]:
        slug = slugify(workspace_name)
        workspace = self.workspace_manager.workspace(slug)
        return workspace.get_chunks(limit=limit)

    def get_chunks_with_embeddings(
        self, workspace_name: str, limit: int = 10_000
    ) -> list[ChunkWithEmbedding]:
        slug = slugify(workspace_name)
        workspace = self.workspace_manager.workspace(slug)
        return workspace.get_chunks_with_embeddings(limit)

    def get_documents(self, workspace_name: str, limit: int = 10_000) -> list[Document]:
        slug = slugify(workspace_name)
        workspace = self.workspace_manager.workspace(slug)
        return workspace.get_documents(limit=limit)

    def delete_document(self, workspace_name: str, document_id: UUID) -> None:
        slug = slugify(workspace_name)
        workspace = self.workspace_manager.workspace(slug)

        doc = workspace.get_document(document_id)
        if not doc:
            return

        ext = Path(doc.filename).suffix.lower().lstrip(".")

        workspace.delete_document(document_id)
        self.object_storage.delete_file(
            workspace=slug, document_id=document_id, extension=ext
        )

    def _persist(
        self,
        workspace: Workspace,
        workspace_slug: str,
        document: Document,
        chunks: list[Chunk],
        embeddings: list[Embedding],
    ) -> None:
        try:
            workspace.upsert_document(document)
            workspace.upsert_chunks(chunks, embeddings)
        except Exception:
            try:
                workspace.delete_document(document.document_id)
            except Exception:
                pass

            try:
                ext = Path(document.filename).suffix.lower().lstrip(".")
                self.object_storage.delete_file(
                    workspace_slug, document.document_id, ext
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
