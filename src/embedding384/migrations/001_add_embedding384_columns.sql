-- Add embedding columns to casts table
ALTER TABLE casts 
ADD COLUMN IF NOT EXISTS embedding vector(384),
ADD COLUMN IF NOT EXISTS embedding_updated_at TIMESTAMP WITH TIME ZONE; 