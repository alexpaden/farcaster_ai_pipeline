-- Create indexes for production casts table
DO $$ 
BEGIN
    -- Index for similarity search
    IF NOT EXISTS (SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
                  WHERE c.relname = 'casts_embedding_idx' AND n.nspname = 'public') THEN
        CREATE INDEX casts_embedding_idx ON public.casts 
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 1000);  -- Larger lists for production
    END IF;

    -- Index for finding unprocessed casts
    IF NOT EXISTS (SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
                  WHERE c.relname = 'idx_casts_embedding_null' AND n.nspname = 'public') THEN
        CREATE INDEX idx_casts_embedding_null ON public.casts (id) 
        WHERE embedding IS NULL;
    END IF;

    -- Index for embedding update tracking
    IF NOT EXISTS (SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
                  WHERE c.relname = 'idx_casts_embedding_updated' AND n.nspname = 'public') THEN
        CREATE INDEX idx_casts_embedding_updated ON public.casts (embedding_updated_at)
        WHERE embedding_updated_at IS NOT NULL;
    END IF;
END $$; 