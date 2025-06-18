-- General unprocessed casts index, optimized for timestamp ordering and fid join
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_casts_threads_pending_ts_fid
  ON farcaster.casts ("timestamp", fid)
  WHERE threads_status = 0;


-- Tiny index (≈500 k rows) to probe the label table
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ul_target_spam
    ON farcaster.user_labels (target_fid)
    WHERE label_type = 'spam';          -- one row per fid


-- Index for fast feed retrieval of NOT-SPAM rows (spam = 2)
-- ORDER BY timestamp ASC/DESC works (Postgres can scan both ways)
-- Add INCLUDE (...) later if you need index-only scans with more columns
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_threads_ts_spam2
    ON unbias.threads ("timestamp")
    WHERE spam = 2;


-- Composite index for batch processing selection
-- This is the most critical index for the process_thread_batches procedure
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_threads_batch_processing
    ON unbias.threads (threads_status, spam, "timestamp")
    WHERE threads_status = 0 AND spam = 2;

-- Index on author_fid for spam label sync and potential filtering
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_threads_author_fid
    ON unbias.threads (author_fid);

-- Index on claimed_at for monitoring/debugging stuck threads
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_threads_claimed_at
    ON unbias.threads (claimed_at)
    WHERE threads_status = 1;  -- Only for claimed threads

-- Index on reactions for potential sorting/filtering by popularity
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_threads_reactions
    ON unbias.threads (reactions DESC)
    WHERE spam = 2;  -- Only for non-spam threads

-- Partial index for processed threads that might be queried
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_threads_processed
    ON unbias.threads ("timestamp" DESC)
    WHERE threads_status = 1 AND spam = 2;

-- Index for threads currently being processed by embedding workers (status = 2)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_threads_embedding_claimed
    ON unbias.threads ("timestamp")
    WHERE threads_status = 2;

-- Index for threads with completed embeddings (status = 3)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_threads_embedding_complete
    ON unbias.threads ("timestamp" DESC)
    WHERE threads_status = 3 and spam = 2;


-- 1. B‑tree on author_fid
CREATE INDEX CONCURRENTLY idx_threads_author_fid
  ON unbias.threads (author_fid);

-- 2. GIN on fids array
CREATE INDEX CONCURRENTLY idx_threads_fids_gin
  ON unbias.threads
  USING GIN (fids);

CREATE INDEX CONCURRENTLY threads_spam2_status_ts_desc_idx
    ON unbias.threads (threads_status, timestamp DESC)
    WHERE spam = 2;

