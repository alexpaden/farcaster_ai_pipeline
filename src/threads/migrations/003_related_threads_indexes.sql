-- 1. Keep - Unprocessed casts index
CREATE INDEX CONCURRENTLY idx_casts_pending_ts_fid
    ON nindexer.casts ("timestamp", fid)
    WHERE threads_status = 0;  -- Or whatever column tracks processing

-- 2. Keep - User spam labels lookup
CREATE INDEX CONCURRENTLY idx_user_labels_spam_target
    ON farcaster.user_labels (target_fid)
    WHERE label_type = 'spam';

-- 3. Keep - Feed retrieval for non-spam threads
CREATE INDEX CONCURRENTLY idx_threads_nonspam_timestamp
    ON unbias.threads ("timestamp" DESC)
    WHERE spam = 2;

-- 4. Keep - Critical batch processing for unprocessed non-spam threads
CREATE INDEX CONCURRENTLY idx_threads_unprocessed_nonspam
    ON unbias.threads (thread_status, spam, "timestamp")
    WHERE thread_status = 0 AND spam = 2;

-- 5. Keep - Author FID lookups (removed duplicate)
CREATE INDEX CONCURRENTLY idx_threads_author
    ON unbias.threads (author_fid);

-- 6. Keep - Popularity-based queries
CREATE INDEX CONCURRENTLY idx_threads_nonspam_reactions
    ON unbias.threads (reactions DESC)
    WHERE spam = 2;

-- 7. Keep - Multi-status queries with spam filter
CREATE INDEX CONCURRENTLY idx_threads_nonspam_status_timestamp
    ON unbias.threads (thread_status, "timestamp" DESC)
    WHERE spam = 2;

-- 8. Keep - FID array searches
CREATE INDEX CONCURRENTLY idx_threads_fids_array
    ON unbias.threads USING GIN (fids);