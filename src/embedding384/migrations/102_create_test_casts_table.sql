-- Create test_casts table for benchmarking
CREATE TABLE IF NOT EXISTS test_casts (
    id BIGINT PRIMARY KEY,
    text TEXT,
    embedding384 vector(384),
    embedding384_updated_at TIMESTAMP WITH TIME ZONE
);

-- Create indexes for test_casts table
CREATE INDEX IF NOT EXISTS idx_test_casts_embedding_cosine 
ON test_casts 
USING ivfflat (embedding384 vector_cosine_ops)
WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_test_casts_unprocessed 
ON test_casts (id) 
WHERE embedding384 IS NULL AND text IS NOT NULL; 