-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Add embedding columns to existing casts table if they don't exist
DO $$ 
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                  WHERE table_schema = 'public' 
                  AND table_name = 'casts'
                  AND column_name = 'embedding') THEN
        ALTER TABLE public.casts ADD COLUMN embedding vector(384);
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                  WHERE table_schema = 'public' 
                  AND table_name = 'casts'
                  AND column_name = 'embedding_updated_at') THEN
        ALTER TABLE public.casts ADD COLUMN embedding_updated_at timestamp;
    END IF;
END $$;

-- Create test_casts table if not exists
CREATE TABLE IF NOT EXISTS public.test_casts (
    id SERIAL PRIMARY KEY,
    cast_id BIGINT NOT NULL,  -- Original cast ID
    text TEXT,
    embedding vector(384),  -- For all-MiniLM-L6-v2 model
    embedding_started_at TIMESTAMP,
    embedding_updated_at TIMESTAMP,
    error_count INTEGER DEFAULT 0,
    last_error TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes if they don't exist
DO $$ 
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
                  WHERE c.relname = 'casts_embedding_idx' AND n.nspname = 'public') THEN
        CREATE INDEX casts_embedding_idx ON public.casts USING ivfflat (embedding vector_cosine_ops);
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
                  WHERE c.relname = 'test_casts_embedding_idx' AND n.nspname = 'public') THEN
        CREATE INDEX test_casts_embedding_idx ON public.test_casts USING ivfflat (embedding vector_cosine_ops);
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
                  WHERE c.relname = 'idx_test_casts_embedding_null' AND n.nspname = 'public') THEN
        CREATE INDEX idx_test_casts_embedding_null ON public.test_casts (id) WHERE embedding IS NULL;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
                  WHERE c.relname = 'idx_test_casts_cast_id' AND n.nspname = 'public') THEN
        CREATE INDEX idx_test_casts_cast_id ON public.test_casts (cast_id);
    END IF;
END $$; 