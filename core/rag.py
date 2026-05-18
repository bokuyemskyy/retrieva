from __future__ import annotations

from typing import Any, Dict, List, Optional

from slugify import slugify

from core.embedding.base_embedder import BaseEmbedder
from core.ingestion.chunker import Chunker
from core.ingestion.ingestor import Ingestor
from core.ingestion.processor.audio_processor import AudioProcessor
from core.ingestion.processor.document_processor import DocumentProcessor
from core.ingestion.processor.image_processor import ImageProcessor
from core.ingestion.processor.text_processor import TextProcessor
from core.retrieval.base_retriever import BaseRetriever
from core.retrieval.qdrant_retriever import QdrantRetriever
from core.storage.qdrant_vector_store import QdrantVectorStore
from core.storage.storage_manager import StorageManager
from core.ingestion.image_captioner import ImageCaptioner


def _build_default_processors(chunker: Chunker) -> Dict[str, Any]:
    image_proc = ImageCaptioner
    return {
        "pdf": DocumentProcessor(chunker=chunker, image_processor=image_proc),
        "txt": TextProcessor(chunker=chunker),
        "md": TextProcessor(chunker=chunker),
        "png": ImageProcessor(chunker=chunker, image_processor=image_proc),
        "jpg": ImageProcessor(chunker=chunker, image_processor=image_proc),
        "jpeg": ImageProcessor(chunker=chunker, image_processor=image_proc),
        "wav": AudioProcessor(chunker=chunker),
        "mp3": AudioProcessor(chunker=chunker),
    }


class RAG:
    def __init__(
        self,
        workspace: str,
        qdrant_url: str,
        storage_dir: str,
        embedder: BaseEmbedder,
        llm_client,
        chunker: Optional[Chunker] = None,
        retriever: Optional[BaseRetriever] = None,
        processors: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
    ):
        self.workspace = slugify(workspace)

        _chunker = chunker or Chunker()

        self.storage = StorageManager(storage_dir, self.workspace)

        vector_size = getattr(embedder, "vector_size", 1536)

        self.vector_store = QdrantVectorStore(
            url=qdrant_url,
            workspace=self.workspace,
            vector_size=vector_size,
        )

        _processors = processors or _build_default_processors(_chunker)

        self.ingestor = Ingestor(
            storage_manager=self.storage,
            embedder=embedder,
            vector_store=self.vector_store,
            chunker=_chunker,
            processors=_processors,
        )

        self.retriever = retriever or QdrantRetriever(
            vector_store=self.vector_store,
            embedder=embedder,
        )

        self.llm_client = llm_client

        self.system_prompt = system_prompt or (
            "Answer the question using only the provided context. "
            "If the context does not contain enough information, say so.\n\n"
            "Context:\n{context}\n\n"
            "Question: {query}"
        )

    def add_file(
        self, path: str, file_type: str, metadata: Optional[dict] = None
    ) -> str:
        return self.ingestor.save_and_process_file(
            source_path=path,
            file_type=file_type,
            metadata=metadata or {},
        )

    def delete_document(self, document_id: str) -> None:
        self.vector_store.delete_document(document_id)

    def retrieve_chunks(
        self, query: str, top_k: int = 5, fetch_k: int = 30
    ) -> List[Dict[str, Any]]:
        return self.retriever.retrieve(query=query, top_k=top_k, fetch_k=fetch_k)

    def generate_response(self, query: str, top_k: int = 5, fetch_k: int = 30) -> str:
        chunks = self.retrieve_chunks(query, top_k=top_k, fetch_k=fetch_k)

        if not chunks:
            return "No relevant information was found in the knowledge base."

        context = "\n\n".join(
            f"[score={c.get('score', 0):.3f}]\n{c['text']}" for c in chunks
        )

        prompt = self.system_prompt.format(query=query, context=context)

        response = self.llm_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
        )

        return response.choices[0].message.content.strip()

    def get_ingested_chunks(self, limit: int = 10_000) -> List[Dict[str, Any]]:
        return self.vector_store.get_all_chunks(limit=limit)

    def get_ingested_documents(self, limit: int = 10_000) -> List[Dict[str, Any]]:
        return self.vector_store.get_all_documents(limit=limit)

    def clear_knowledge_base(self) -> None:
        self.vector_store.delete_all_chunks_and_documents()
        self.storage.clear_all_files()

    def delete_workspace(self) -> None:
        self.vector_store.delete_workspace_collections()
        self.storage.delete_workspace_directory()


def default_rag_client(
    llm_client,
    embedder: BaseEmbedder,
    qdrant_url: str = "http://localhost:6333",
    storage_dir: str = "./storage",
    workspace: str = "default",
) -> RAG:
    return RAG(
        workspace=workspace,
        qdrant_url=qdrant_url,
        storage_dir=storage_dir,
        embedder=embedder,
        llm_client=llm_client,
    )
