CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    document_id UUID PRIMARY KEY,
    filename TEXT NOT NULL,
    workspace TEXT NOT NULL,
    source_path TEXT NOT NULL,
    original_path TEXT NOT NULL,
    content_hash TEXT NOT NULL, 
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id UUID PRIMARY KEY,
    workspace TEXT NOT NULL,
    document_id UUID NOT NULL,
    content TEXT NOT NULL,
    modality TEXT NOT NULL,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding VECTOR(1024) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    CONSTRAINT fk_document
        FOREIGN KEY(document_id)
        REFERENCES documents(document_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_documents_workspace
ON documents(workspace);

CREATE INDEX IF NOT EXISTS idx_documents_content_hash 
ON documents(content_hash);

CREATE INDEX IF NOT EXISTS idx_chunks_workspace
ON chunks(workspace);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id
ON chunks(document_id);

CREATE INDEX IF NOT EXISTS idx_chunks_embedding
ON chunks
USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_chunks_text_search
ON chunks
USING GIN (to_tsvector('english', content));