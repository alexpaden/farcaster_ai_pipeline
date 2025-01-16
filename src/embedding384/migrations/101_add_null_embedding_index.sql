-- Add partial index for efficient null embedding lookups
CREATE INDEX IF NOT EXISTS idx_casts_null_embedding384 
ON public.casts (id)
WHERE embedding384 IS NULL; 