-- Add index optimized for fast counting of processed/unprocessed rows
-- This partial index will be very small and fast since it only stores a boolean and id
-- The WHERE clause matches our query conditions exactly
CREATE INDEX IF NOT EXISTS idx_casts_embedding_count 
ON public.casts ((embedding384 IS NULL))
INCLUDE (id)
WHERE text IS NOT NULL 
AND length(trim(text)) > 0;

-- Update statistics for the new index
ANALYZE public.casts; 