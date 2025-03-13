-- Create function to encapsulate optimized count query
CREATE OR REPLACE FUNCTION unbias.get_unprocessed_count()
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
            FROM farcaster.casts 
            WHERE embedding384 IS NULL 
            AND text IS NOT NULL 
            AND length(trim(text)) > 0
            FOR UPDATE SKIP LOCKED
        ) sub
    );
END;
$$ LANGUAGE plpgsql; 