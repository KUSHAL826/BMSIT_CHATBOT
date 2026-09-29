-- Supabase / Render PostgreSQL Schema with pgvector
-- Run this script in Supabase Dashboard -> SQL Editor -> New Query

-- 1. Enable the pgvector extension for AI vector search
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Create the vectors table for storing text chunks and embeddings
CREATE TABLE IF NOT EXISTS vectors (
    id TEXT PRIMARY KEY,
    source_id TEXT,
    source_name TEXT,
    source_type TEXT,
    item_key TEXT,
    text TEXT NOT NULL,
    tokens INT DEFAULT 0,
    context_header TEXT,
    metadata JSONB DEFAULT '{}'::jsonb,
    embed_provider TEXT,
    content_hash TEXT,
    embedding vector(768),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 3. Create an IVFFLAT vector index for high-speed similarity search
CREATE INDEX IF NOT EXISTS vectors_embedding_idx ON vectors 
USING ivfflat (embedding vector_ip_ops) WITH (lists = 100);
