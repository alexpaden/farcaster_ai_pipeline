-- Create optimized index for batch processing that handles concurrency
CREATE INDEX IF NOT EXISTS idx_casts_batch_processing
ON public.casts (id)  -- Index on id for ORDER BY and range conditions
INCLUDE (text, embedding384_updated_at)  -- Include text and updated_at to avoid table lookups
WHERE embedding384 IS NULL 
AND text IS NOT NULL 
AND length(trim(text)) > 0;

-- Add index for finding and cleaning up stale processing
CREATE INDEX IF NOT EXISTS idx_casts_stale_processing
ON public.casts (embedding384_updated_at)
WHERE embedding384 IS NULL 
AND embedding384_updated_at IS NOT NULL;

-- Add index statistics hint for better query planning
ANALYZE public.casts; 