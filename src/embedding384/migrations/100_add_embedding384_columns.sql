-- Add vector column for embeddings
ALTER TABLE public.casts
ADD COLUMN IF NOT EXISTS embedding384 vector(384); 

ALTER TABLE public.casts
ADD COLUMN IF NOT EXISTS embedding384_updated_at TIMESTAMP WITH TIME ZONE;