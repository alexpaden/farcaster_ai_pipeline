-- Add vector column for embeddings
ALTER TABLE public.casts
ADD COLUMN IF NOT EXISTS embedding384 vector(384); 