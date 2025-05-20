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


