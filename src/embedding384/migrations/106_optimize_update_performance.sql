-- Create partial index to optimize UPDATE operations and reduce lock contention
CREATE INDEX IF NOT EXISTS idx_casts_embedding_updates
ON public.casts (id)
INCLUDE (embedding384_updated_at)
WHERE embedding384 IS NULL;

-- Add index statistics hint for better query planning
ANALYZE public.casts;

-- Create function to handle updates with better concurrency
CREATE OR REPLACE FUNCTION public.batch_update_embeddings(
    p_ids bigint[],
    p_embeddings text[]  -- Changed to text[] to accept formatted vectors
) RETURNS void AS $$
BEGIN
    -- Set a reasonable statement timeout
    SET LOCAL statement_timeout = '60s';
    SET LOCAL lock_timeout = '30s';
    
    -- Use READ COMMITTED to reduce lock contention
    SET LOCAL TRANSACTION ISOLATION LEVEL READ COMMITTED;
    
    -- Single bulk update for the entire batch
    UPDATE public.casts t
    SET embedding384 = c.embedding::vector,
        embedding384_updated_at = NOW()
    FROM (
        SELECT unnest(p_ids) as id,
               unnest(p_embeddings)::vector as embedding
    ) c
    WHERE t.id = c.id
    AND (t.embedding384 IS NULL OR t.embedding384_updated_at < NOW() - interval '1 hour');
END;
$$ LANGUAGE plpgsql; 