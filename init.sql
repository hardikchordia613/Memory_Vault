-- Enable UUID generation and pgvector extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;

-- Create the core memory storage table
CREATE TABLE IF NOT EXISTS code_memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    file_path VARCHAR(512),
    developer_context TEXT NOT NULL,
    raw_code TEXT NOT NULL,
    embedding VECTOR(768) NOT NULL,
    search_vector tsvector GENERATED ALWAYS AS (
        to_tsvector('english', developer_context || ' ' || raw_code)
    ) STORED,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Upgrade databases whose code_memories table predates hybrid search.
ALTER TABLE code_memories
ADD COLUMN IF NOT EXISTS search_vector tsvector GENERATED ALWAYS AS (
    to_tsvector('english', developer_context || ' ' || raw_code)
) STORED;

-- Create HNSW index for high-performance cosine distance similarity search
-- vector_cosine_ops matches the <=> cosine distance operator
CREATE INDEX IF NOT EXISTS idx_code_memories_embedding 
ON code_memories 
USING hnsw (embedding vector_cosine_ops);

-- Accelerate exact keyword/full-text retrieval.
CREATE INDEX IF NOT EXISTS idx_code_memories_search_vector
ON code_memories
USING GIN (search_vector);
