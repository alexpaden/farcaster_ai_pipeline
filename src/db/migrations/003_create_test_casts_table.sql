-- Create test_casts table for benchmarking and testing
CREATE TABLE IF NOT EXISTS public.test_casts (
    id SERIAL PRIMARY KEY,
    cast_id BIGINT NOT NULL,
    text TEXT,
    embedding vector(384),
    embedding_started_at TIMESTAMP,
    embedding_updated_at TIMESTAMP,
    error_count INTEGER DEFAULT 0,
    last_error TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for test_casts table
DO $$ 
BEGIN
    -- Index for similarity search
    IF NOT EXISTS (SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
                  WHERE c.relname = 'test_casts_embedding_idx' AND n.nspname = 'public') THEN
        CREATE INDEX test_casts_embedding_idx ON public.test_casts 
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100);  -- Smaller lists for test table
    END IF;

    -- Index for finding unprocessed casts
    IF NOT EXISTS (SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
                  WHERE c.relname = 'idx_test_casts_embedding_null' AND n.nspname = 'public') THEN
        CREATE INDEX idx_test_casts_embedding_null ON public.test_casts (id) 
        WHERE embedding IS NULL;
    END IF;

    -- Index for looking up by original cast_id
    IF NOT EXISTS (SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
                  WHERE c.relname = 'idx_test_casts_cast_id' AND n.nspname = 'public') THEN
        CREATE INDEX idx_test_casts_cast_id ON public.test_casts (cast_id);
    END IF;
END $$; 