from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from uuid import UUID
from core.embedding.embedder import BaseEmbedder
from core.models import Chunk, Document
from sqlalchemy import (
    String,
    Integer,
    Text,
    ForeignKey,
    create_engine,
    schema,
    text,
    MetaData,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from pgvector.sqlalchemy import Vector  # type: ignore


class SystemBase(DeclarativeBase):
    pass


class WorkspaceRegistry(SystemBase):
    __tablename__ = "workspaces"

    workspace_name: Mapped[str] = mapped_column(String, primary_key=True)
    schema_name: Mapped[str] = mapped_column(String, unique=True, nullable=False)

    vector_size: Mapped[int] = mapped_column(Integer, nullable=False)
    model_name: Mapped[str] = mapped_column(String, nullable=False)
    model_provider: Mapped[str] = mapped_column(String, nullable=False)

    config: Mapped[dict] = mapped_column(JSONB, server_default=text("'{}'::jsonb"))


@dataclass
class WorkspaceConfig:
    name: str
    embedding_provider: str
    embedding_model_name: str
    vector_size: int
    embedding_api_key: Optional[str] = None
    embedding_base_url: Optional[str] = None


_MODEL_CACHE: Dict[str, Any] = {}


def _get_workspace_models(schema_name: str, vector_size: int):
    if schema_name in _MODEL_CACHE:
        return _MODEL_CACHE[schema_name]

    workspace_metadata = MetaData(schema=schema_name)

    class WorkspaceBase(DeclarativeBase):
        metadata = workspace_metadata

    class DocumentModel(WorkspaceBase):
        __tablename__ = "documents"
        document_id: Mapped[UUID] = mapped_column(
            PG_UUID(as_uuid=True), primary_key=True
        )
        filename: Mapped[str] = mapped_column(Text, nullable=False)
        source_path: Mapped[str] = mapped_column(Text, nullable=False)
        original_path: Mapped[str] = mapped_column(Text, nullable=False)
        content_hash: Mapped[str] = mapped_column(Text, nullable=False, index=True)

    class ChunkModel(WorkspaceBase):
        __tablename__ = "chunks"
        chunk_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True)
        document_id: Mapped[UUID] = mapped_column(
            PG_UUID(as_uuid=True),
            ForeignKey("documents.document_id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        )
        content: Mapped[str] = mapped_column(Text, nullable=False)
        modality: Mapped[str] = mapped_column(Text, nullable=False)
        metadata_json: Mapped[dict[str, Any]] = mapped_column(
            JSONB, server_default=text("'{}'::jsonb")
        )
        embedding: Mapped[List[float]] = mapped_column(
            Vector(vector_size), nullable=False
        )

    _MODEL_CACHE[schema_name] = (WorkspaceBase, DocumentModel, ChunkModel)
    return _MODEL_CACHE[schema_name]


class Workspace:
    def __init__(self, session_maker, schema_name: str, config: WorkspaceConfig):
        self.SessionLocal = session_maker
        self.schema_name = schema_name
        self.config = config

        _, self.DocumentModel, self.ChunkModel = _get_workspace_models(
            schema_name, config.vector_size
        )

    def upsert_document(self, doc: Document) -> None:
        with self.SessionLocal() as session:
            existing = session.get(
                self.DocumentModel, doc.document_id
            ) or self.DocumentModel(document_id=doc.document_id)
            existing.filename = doc.filename
            existing.source_path = doc.source_path
            existing.original_path = doc.original_path
            existing.content_hash = doc.content_hash
            session.add(existing)
            session.commit()

    def upsert_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("The number of chunks and embeddings must be identical.")

        with self.SessionLocal() as session:
            for chunk, embedding in zip(chunks, embeddings):
                if len(embedding) != self.config.vector_size:
                    raise ValueError(
                        f"Dim mismatch for {chunk.chunk_id}. "
                        f"Expected {self.config.vector_size}, got {len(embedding)}"
                    )

                existing = session.get(
                    self.ChunkModel, chunk.chunk_id
                ) or self.ChunkModel(chunk_id=chunk.chunk_id)

                existing.document_id = chunk.document_id
                existing.content = chunk.content
                existing.modality = chunk.modality
                existing.metadata_json = chunk.metadata
                existing.embedding = embedding

                session.add(existing)
            session.commit()

    def search(self, query_vector: List[float], top_k: int = 5) -> List[dict]:
        with self.SessionLocal() as session:
            distance = self.ChunkModel.embedding.cosine_distance(query_vector).label(
                "distance"
            )
            query = (
                session.query(self.ChunkModel, distance)
                .order_by(distance.asc())
                .limit(top_k)
            )

            return [
                {
                    "chunk_id": row.chunk_id,
                    "content": row.content,
                    "score": 1.0 - float(dist),
                }
                for row, dist in query.all()
            ]

    def get_document_by_hash(self, content_hash: str) -> Optional[UUID]:
        with self.SessionLocal() as session:
            doc = (
                session.query(self.DocumentModel)
                .filter(self.DocumentModel.content_hash == content_hash)
                .first()
            )

            return doc.document_id if doc else None

    def get_documents(self, limit: int = 1000) -> List[Document]:
        with self.SessionLocal() as session:
            rows = (
                session.query(self.DocumentModel)
                .order_by(self.DocumentModel.document_id)
                .limit(limit)
                .all()
            )

            return [
                Document(
                    document_id=row.document_id,
                    filename=row.filename,
                    source_path=row.source_path,
                    original_path=row.original_path,
                    content_hash=row.content_hash,
                )
                for row in rows
            ]

    def get_document(self, document_id: UUID) -> Optional[Document]:
        with self.SessionLocal() as session:
            row = session.get(self.DocumentModel, document_id)
            if row:
                return Document(
                    document_id=row.document_id,
                    filename=row.filename,
                    source_path=row.source_path,
                    original_path=row.original_path,
                    content_hash=row.content_hash,
                )
            return None

    def delete_document(self, document_id: UUID) -> None:
        with self.SessionLocal() as session:
            row = session.get(self.DocumentModel, document_id)
            if row:
                session.delete(row)
                session.commit()

    def get_chunks(self, limit: int = 1000) -> List[Chunk]:
        with self.SessionLocal() as session:
            rows = (
                session.query(self.ChunkModel)
                .order_by(self.ChunkModel.chunk_id)
                .limit(limit)
                .all()
            )
            return [
                Chunk(
                    chunk_id=row.chunk_id,
                    document_id=row.document_id,
                    content=row.content,
                    modality=row.modality,
                    metadata=row.metadata_json,
                )
                for row in rows
            ]


class VectorStorage:
    def __init__(self, db_url: str):
        self.engine = create_engine(db_url)
        self.SessionLocal = sessionmaker(
            bind=self.engine, autoflush=False, autocommit=False
        )

        with self.engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        SystemBase.metadata.create_all(self.engine)

    def create_workspace(
        self,
        name: str,
        embedder: BaseEmbedder,
        config: Optional[dict] = None,
    ) -> Workspace:
        schema_name = f"ws_{name}"
        config = config or {}

        with self.SessionLocal() as session:
            if not session.get(WorkspaceRegistry, name):
                reg = WorkspaceRegistry(
                    workspace_name=name,
                    schema_name=schema_name,
                    vector_size=embedder.vector_size,
                    model_name=embedder.model_name,
                    model_provider=embedder.provider,
                    config=config,
                )
                session.add(reg)
                session.commit()

        with self.engine.begin() as conn:
            conn.execute(schema.CreateSchema(schema_name, if_not_exists=True))

        WorkspaceBase, _, _ = _get_workspace_models(schema_name, embedder.vector_size)
        WorkspaceBase.metadata.create_all(self.engine)

        return self.workspace(name)

    def workspace(self, name: str) -> Workspace:
        with self.SessionLocal() as session:
            reg = session.get(WorkspaceRegistry, name)
            if not reg:
                raise ValueError(f"Workspace '{name}' not found.")

            config = WorkspaceConfig(
                name=reg.workspace_name,
                embedding_provider=reg.model_provider,
                embedding_model_name=reg.model_name,
                vector_size=reg.vector_size,
                embedding_api_key=reg.config.get("api_key"),
                embedding_base_url=reg.config.get("base_url"),
            )
            return Workspace(self.SessionLocal, reg.schema_name, config)

    def delete_workspace(self, name: str) -> None:
        with self.SessionLocal() as session:
            reg = session.get(WorkspaceRegistry, name)
            if reg:
                with self.engine.begin() as conn:
                    conn.execute(
                        text(f"DROP SCHEMA IF EXISTS {reg.schema_name} CASCADE;")
                    )
                session.delete(reg)
                session.commit()

    def get_workspaces(self) -> List[WorkspaceConfig]:
        with self.SessionLocal() as session:
            registries = session.query(WorkspaceRegistry).all()
            return [
                WorkspaceConfig(
                    name=reg.workspace_name,
                    embedding_provider=reg.model_provider,
                    embedding_model_name=reg.model_name,
                    vector_size=reg.vector_size,
                    embedding_api_key=reg.config.get("api_key"),
                    embedding_base_url=reg.config.get("base_url"),
                )
                for reg in registries
            ]

    def delete_all_workspaces(self) -> None:
        with self.SessionLocal() as session:
            registries = session.query(WorkspaceRegistry).all()

            with self.engine.begin() as conn:
                for reg in registries:
                    conn.execute(
                        text(f"DROP SCHEMA IF EXISTS {reg.schema_name} CASCADE;")
                    )

            session.query(WorkspaceRegistry).delete()
            session.commit()
