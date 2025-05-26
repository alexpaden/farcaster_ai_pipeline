-- Index for fast lookup of unclaimed media (media_status = 0)
CREATE INDEX CONCURRENTLY idx_media_status_unclaimed
  ON unbias.media(media_status)
  WHERE media_status = 0;

-- Index for farcaster.process_media_casts procedure with fid for user_labels join
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_casts_media_pending_fid_ts_id
  ON farcaster.casts (fid, "timestamp", id)
  WHERE media_status = 0 AND embeds <> '"[]"';

  -- delete this index later, testing onyl.
CREATE INDEX CONCURRENTLY idx_casts_media_nonzero_id
  ON farcaster.casts (id)
  WHERE media_status <> 0;

CREATE INDEX CONCURRENTLY idx_casts_threads_pending_ts_fid_ccnew1
 ON farcaster.casts("timestamp", fid) 
 WHERE (threads_status = 0);