-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Test casts table for benchmarking
DROP TABLE IF EXISTS public.test_casts;
CREATE TABLE public.test_casts (
    cast_id bigint PRIMARY KEY,
    text text NOT NULL,
    embedding vector(384)  -- MiniLM-L6-v2 embedding size
);

-- Production casts table with embeddings
ALTER TABLE public.casts ADD COLUMN IF NOT EXISTS embedding vector(384);

-- Create index for fast similarity search
CREATE INDEX IF NOT EXISTS casts_embedding_idx ON public.casts USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 1000);  -- Adjust lists based on data size

-- Create index for test table
CREATE INDEX IF NOT EXISTS test_casts_embedding_idx ON public.test_casts USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);  -- Smaller for test table 