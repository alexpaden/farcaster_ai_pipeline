-- Add embedding columns to existing casts table
DO $$ 
BEGIN
    -- Add embedding column if it doesn't exist
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                  WHERE table_schema = 'public' 
                  AND table_name = 'casts'
                  AND column_name = 'embedding') THEN
        ALTER TABLE public.casts ADD COLUMN embedding vector(384);
    END IF;

    -- Add timestamp column for tracking updates
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                  WHERE table_schema = 'public' 
                  AND table_name = 'casts'
                  AND column_name = 'embedding_updated_at') THEN
        ALTER TABLE public.casts ADD COLUMN embedding_updated_at timestamp;
    END IF;
END $$; 