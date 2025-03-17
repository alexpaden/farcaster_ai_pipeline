CREATE INDEX idx_casts_unprocessed ON farcaster.casts (id)
INCLUDE (embedding384_updated_at)
WHERE embedding384 IS NULL 
  AND text IS NOT NULL 
  AND text != ''

--  Index for COUNT queries
CREATE INDEX idx_optimized_count ON farcaster.casts ((1)) 
WITH (fillfactor=100)
WHERE embedding384 IS NULL 
  AND text IS NOT NULL 
  AND text != '';

-- 3) Index for processed rows lookups
CREATE INDEX IF NOT EXISTS idx_casts_processed
ON farcaster.casts (id)
WHERE embedding384 IS NOT NULL 

-- 4) Index for stale processing rows (started but never finished)
CREATE INDEX IF NOT EXISTS idx_casts_stale_processing
ON farcaster.casts (embedding384_updated_at)
WHERE embedding384 IS NULL 
  AND embedding384_updated_at IS NOT NULL
  AND text IS NOT NULL
  AND text != '';
