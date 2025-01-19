-- Create a more efficient covering index specifically for count queries
-- Using (1) constant reduces index size since we only need existence info
CREATE INDEX IF NOT EXISTS idx_casts_embedding_count_v2 
ON public.casts ((1))
WHERE embedding384 IS NULL 
AND text IS NOT NULL 
AND length(trim(text)) > 0;

-- Update statistics to help query planner
ANALYZE public.casts;

-- Create function to encapsulate optimized count query
CREATE OR REPLACE FUNCTION get_unprocessed_count()
RETURNS bigint AS $$
BEGIN
    -- Set READ COMMITTED isolation to ensure we don't use stale data
    SET LOCAL TRANSACTION ISOLATION LEVEL READ COMMITTED;
    
    -- Temporarily disable bitmap and sequential scans to force index usage
    SET LOCAL enable_bitmapscan = off;
    SET LOCAL enable_seqscan = off;
    
    RETURN (
        SELECT count(*) 
        FROM (
            SELECT 1 
            FROM public.casts 
            WHERE embedding384 IS NULL 
            AND text IS NOT NULL 
            AND length(trim(text)) > 0
        ) sub
    );
END;
$$ LANGUAGE plpgsql; 