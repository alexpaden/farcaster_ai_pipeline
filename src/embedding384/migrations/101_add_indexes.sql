-- OPTIMIZED PRIMARY INDEXES FOR MAIN QUERIES

-- 1) Primary index for finding unprocessed rows efficiently
CREATE INDEX IF NOT EXISTS idx_casts_unprocessed
ON farcaster.casts (id)
INCLUDE (text, embedding384_updated_at)
WHERE embedding384 IS NULL
  AND text IS NOT NULL
  AND LENGTH(TRIM(text)) > 0;

-- 2) Index for counting unprocessed rows
CREATE INDEX IF NOT EXISTS idx_casts_embedding_count 
ON farcaster.casts ((1))
WHERE embedding384 IS NULL 
AND text IS NOT NULL 
AND length(trim(text)) > 0;

-- 3) Index for processed rows lookups
CREATE INDEX IF NOT EXISTS idx_casts_processed
ON farcaster.casts (id)
WHERE embedding384 IS NOT NULL 
AND LENGTH(TRIM(text)) > 0;

-- Remove everything else as they're redundant