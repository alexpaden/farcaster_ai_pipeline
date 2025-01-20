-- Add vector column for embeddings
ALTER TABLE public.casts
ADD COLUMN IF NOT EXISTS embedding384 vector(384); 

ALTER TABLE public.casts
ADD COLUMN IF NOT EXISTS embedding384_updated_at TIMESTAMP WITH TIME ZONE;


-- Create test_casts table for benchmarking
CREATE TABLE IF NOT EXISTS test_casts (
    id BIGINT PRIMARY KEY,
    text TEXT,
    embedding384 vector(384),
    embedding384_updated_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX IF NOT EXISTS idx_test_casts_unprocessed 
ON test_casts (id) 
WHERE embedding384 IS NULL AND text IS NOT NULL; 