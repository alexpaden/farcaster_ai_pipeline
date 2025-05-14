CREATE INDEX CONCURRENTLY idx_casts_root_parent_hash
    ON farcaster.casts (root_parent_hash);

CREATE INDEX CONCURRENTLY idx_casts_parent_hash
    ON farcaster.casts (parent_hash);

CREATE INDEX CONCURRENTLY idx_reactions_fid_target_hash 
  ON farcaster.reactions (fid, target_hash);

CREATE INDEX CONCURRENTLY idx_casts_created_at
  ON farcaster.casts (created_at);

-- Index to quickly find unprocessed root casts (status = 2), ordered by creation time
-- NOTE: This index is potentially redundant if using an all-in-one query that skips status 2.
-- It IS required for the simpler, recommended two-step approach (0->1/2, then 2->3 + Insert).
-- DROP INDEX CONCURRENTLY IF EXISTS idx_casts_status_2_unprocessed_root;
-- CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_casts_status_2_unprocessed_root 
--   ON farcaster.casts (created_at) 
--   WHERE threads_status = 2;

-- Index for user_labels table to optimize joins and filtering in unbias.reaction_counts view
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_labels_target_fid_label_value 
  ON farcaster.user_labels (target_fid, label_value);

-- Index for casts table to optimize joins in unbias.reaction_counts view (comment_counts CTE)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_casts_fid 
  ON farcaster.casts (fid);


-- ##############################################################################
-- # Indexes related to processing initial threads_status = 0
-- ##############################################################################
-- NOTE: The following three indexes are designed to speed up finding rows with
-- threads_status = 0. However, EXPLAIN results have shown the planner may opt
-- for a Sequential Scan, especially with LIMIT clauses or potentially for the
-- full query depending on statistics and cost estimations.
-- The usefulness of these indexes (especially the more specific _and_is_comment
-- and _and_is_root ones) depends heavily on whether the chosen query plan
-- (e.g., using a UNION ALL structure) actually leverages them.
-- REVIEW the EXPLAIN plan for the final backfill query and REMOVE any of these
-- three indexes that are not being used to reduce write overhead and storage.
-- ##############################################################################

-- General index
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_casts_status_0_initial 
  ON farcaster.casts ("timestamp")
  WHERE threads_status = 0;

-- Comment index
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_casts_status_0_and_is_comment 
  ON farcaster.casts ("timestamp")
  WHERE threads_status = 0 AND hash != root_parent_hash;

-- Root index
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_casts_status_0_and_is_root 
  ON farcaster.casts ("timestamp")
  WHERE threads_status = 0 AND hash = root_parent_hash;
