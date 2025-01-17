-- Add partial index for efficient null embedding lookups
CREATE INDEX IF NOT EXISTS idx_casts_null_embedding384 
ON public.casts (id)
WHERE embedding384 IS NULL; 

-- Add index to optimize text filtering conditions
CREATE INDEX IF NOT EXISTS idx_casts_text_processing 
ON public.casts (id)
WHERE text IS NOT NULL AND length(trim(text)) > 0; 

CREATE INDEX IF NOT EXISTS idx_casts_needs_processing
ON public.casts (id)
WHERE embedding384 IS NULL AND LENGTH(TRIM(text)) > 0;

CREATE INDEX IF NOT EXISTS idx_casts_processed
ON public.casts (id)
WHERE embedding384 IS NOT NULL AND LENGTH(TRIM(text)) > 0;

-- Add optimized index for unprocessed row selection and counting
CREATE INDEX IF NOT EXISTS idx_casts_needs_processing
ON public.casts (id)  -- Index on id for ORDER BY and efficient joins
INCLUDE (text)  -- Include text to avoid table lookups
WHERE embedding384 IS NULL 
AND text IS NOT NULL 
AND length(trim(text)) > 0
AND embedding384_updated_at IS NULL;

-- Add index to help find stale processing rows
CREATE INDEX IF NOT EXISTS idx_casts_stale_processing
ON public.casts (embedding384_updated_at)
WHERE embedding384 IS NULL; 