from __future__ import annotations

from typing import Any, List, Optional
from datetime import datetime
from uuid import UUID

from sqlalchemy import ForeignKey, Text, TIMESTAMP, create_engine, func, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Mapped, mapped_column
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from core.models import (
    Chunk,
    ChunkRecord,
    Document,
    DocumentRecord,
    SearchResult,
    TextSearchResult,
)


VECTOR_SIZE = 1024


class Base(DeclarativeBase):
    pass


class DocumentModel(Base):
    __tablename__ = "documents"
    document_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True)
    filename: Mapped[str] = mapped_column(Text, nullable=False)
    workspace: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    source_path: Mapped[str] = mapped_column(Text, nullable=False)
    original_path: Mapped[str] = mapped_column(Text, nullable=False)

    content_hash: Mapped[str] = mapped_column(Text, nullable=False, index=True)

    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP, server_default=func.now(), nullable=False
    )


class ChunkModel(Base):
    __tablename__ = "chunks"

    chunk_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
    )

    workspace: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        index=True,
    )

    document_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("documents.document_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    modality: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'{}'::jsonb"),
    )

    embedding: Mapped[List[float]] = mapped_column(
        Vector(VECTOR_SIZE),
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP,
        server_default=func.now(),
        nullable=False,
    )


class PostgresVectorStorage:
    def __init__(self, db_url: str):

        self.engine = create_engine(db_url)

        self.SessionLocal = sessionmaker(
            bind=self.engine,
            autoflush=False,
            autocommit=False,
        )

        Base.metadata.create_all(self.engine)

    def upsert_document(self, document: Document) -> None:
        with self.SessionLocal() as session:
            existing = session.get(DocumentModel, document.document_id)
            if existing is None:
                existing = DocumentModel(document_id=document.document_id)
                session.add(existing)

            existing.filename = document.filename
            existing.workspace = document.workspace
            existing.source_path = document.source_path
            existing.original_path = document.original_path
            existing.content_hash = document.content_hash

            session.commit()

    def upsert_chunks(self, chunks: List[Chunk]) -> None:
        with self.SessionLocal() as session:
            for chunk in chunks:
                if chunk.document_id is None:
                    raise ValueError(f"Chunk {chunk.chunk_id} missing document_id")

                if chunk.embedding is None:
                    raise ValueError(f"Chunk {chunk.chunk_id} missing embedding")

                if len(chunk.embedding) != VECTOR_SIZE:
                    raise ValueError(
                        f"Chunk {chunk.chunk_id} embedding size "
                        f"{len(chunk.embedding)} != {VECTOR_SIZE}"
                    )

                existing = session.get(
                    ChunkModel,
                    chunk.chunk_id,
                )

                if existing is None:
                    existing = ChunkModel(
                        chunk_id=chunk.chunk_id,
                    )

                    session.add(existing)

                existing.workspace = chunk.workspace
                existing.document_id = chunk.document_id
                existing.content = chunk.content
                existing.modality = chunk.modality.value
                existing.metadata_json = chunk.metadata
                existing.embedding = chunk.embedding

            session.commit()

    def search(
        self,
        query_vector: List[float],
        top_k: int,
        workspace: Optional[str] = None,
    ) -> List[SearchResult]:

        if len(query_vector) != VECTOR_SIZE:
            raise ValueError(f"query_vector size {len(query_vector)} != {VECTOR_SIZE}")

        with self.SessionLocal() as session:
            distance = ChunkModel.embedding.cosine_distance(query_vector).label(
                "distance"
            )

            query = session.query(
                ChunkModel,
                distance,
            )

            if workspace:
                query = query.filter(ChunkModel.workspace == workspace)

            rows = query.order_by(distance.asc()).limit(top_k).all()

            return [
                SearchResult(
                    chunk_id=row.chunk_id,
                    document_id=row.document_id,
                    content=row.content,
                    metadata=row.metadata_json,
                    score=1.0 - float(distance),
                )
                for row, distance in rows
            ]

    def text_search(
        self,
        query_text: str,
        workspace: Optional[str] = None,
        limit: int = 10,
    ) -> List[TextSearchResult]:

        with self.SessionLocal() as session:
            ts_query = func.plainto_tsquery(
                "english",
                query_text,
            )

            rank = func.ts_rank_cd(
                func.to_tsvector(
                    "english",
                    ChunkModel.content,
                ),
                ts_query,
            ).label("rank")

            query = session.query(
                ChunkModel,
                rank,
            )

            if workspace:
                query = query.filter(ChunkModel.workspace == workspace)

            query = query.filter(
                func.to_tsvector(
                    "english",
                    ChunkModel.content,
                ).op("@@")(ts_query)
            )

            rows = query.order_by(rank.desc()).limit(limit).all()

            return [
                TextSearchResult(
                    chunk_id=row.chunk_id,
                    document_id=row.document_id,
                    content=row.content,
                    metadata=row.metadata_json,
                    rank=float(rank),
                )
                for row, rank in rows
            ]

    def get_chunks(
        self,
        workspace: Optional[str] = None,
        limit: int = 1000,
    ) -> List[ChunkRecord]:

        with self.SessionLocal() as session:
            query = session.query(ChunkModel)

            if workspace:
                query = query.filter(ChunkModel.workspace == workspace)

            rows = query.order_by(ChunkModel.created_at.desc()).limit(limit).all()

            return [
                ChunkRecord(
                    chunk_id=row.chunk_id,
                    document_id=row.document_id,
                    workspace=row.workspace,
                    content=row.content,
                    metadata=row.metadata_json,
                )
                for row in rows
            ]

    def get_documents(
        self,
        workspace: Optional[str] = None,
        limit: int = 1000,
    ) -> List[DocumentRecord]:

        with self.SessionLocal() as session:
            query = session.query(DocumentModel)

            if workspace:
                query = query.filter(DocumentModel.workspace == workspace)

            rows = query.order_by(DocumentModel.created_at.desc()).limit(limit).all()

            return [
                DocumentRecord(
                    document_id=row.document_id,
                    workspace=row.workspace,
                    filename=row.filename,
                    source_path=row.source_path,
                    original_path=row.original_path,
                    content_hash=row.content_hash,
                )
                for row in rows
            ]

    def get_document_by_hash(self, workspace: str, content_hash: str) -> Optional[UUID]:
        with self.SessionLocal() as session:
            doc = (
                session.query(DocumentModel)
                .filter_by(workspace=workspace, content_hash=content_hash)
                .first()
            )
            return doc.document_id if doc else None

    def delete_document(self, document_id: UUID) -> None:

        with self.SessionLocal() as session:
            doc = session.get(
                DocumentModel,
                document_id,
            )

            if doc is not None:
                session.delete(doc)
                session.commit()

    def delete_workspace(self, workspace: str) -> None:

        with self.SessionLocal() as session:
            (
                session.query(ChunkModel)
                .filter(ChunkModel.workspace == workspace)
                .delete(synchronize_session=False)
            )

            (
                session.query(DocumentModel)
                .filter(DocumentModel.workspace == workspace)
                .delete(synchronize_session=False)
            )

            session.commit()

    def delete_all_workspaces(self) -> None:

        with self.SessionLocal() as session:
            (session.query(ChunkModel).delete(synchronize_session=False))

            (session.query(DocumentModel).delete(synchronize_session=False))

            session.commit()
