CREATE INDEX CONCURRENTLY idx_casts_root_parent_hash
    ON farcaster.casts (root_parent_hash);

CREATE INDEX CONCURRENTLY idx_casts_parent_hash
    ON farcaster.casts (parent_hash);

CREATE INDEX CONCURRENTLY idx_reactions_fid_target_hash 
  ON farcaster.reactions (fid, target_hash);

CREATE INDEX CONCURRENTLY idx_casts_created_at
  ON farcaster.casts (created_at);

-- Index to quickly find all casts in the initial state (status = 0)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_casts_status_0_initial 
  ON farcaster.casts (created_at) -- Order by creation time, might be useful
  WHERE threads_status = 0;

-- Index to quickly find unprocessed root casts (status = 2), ordered by creation time
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_casts_status_2_unprocessed_root 
  ON farcaster.casts (created_at) 
  WHERE threads_status = 2;

-- Index for user_labels table to optimize joins and filtering in unbias.reaction_counts view
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_labels_target_fid_label_value 
  ON farcaster.user_labels (target_fid, label_value);

-- Index for casts table to optimize joins in unbias.reaction_counts view (comment_counts CTE)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_casts_fid 
  ON farcaster.casts (fid);
