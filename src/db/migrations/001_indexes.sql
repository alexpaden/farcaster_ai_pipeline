-- 1. NEEDS UPDATE: farcaster.casts → nindexer.casts
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_casts_root_parent_hash
    ON nindexer.casts (root_parent_hash);

-- 2. NEEDS UPDATE: farcaster.casts → nindexer.casts
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_casts_parent_hash
    ON nindexer.casts (parent_hash);

-- 3. NEEDS UPDATE: farcaster.reactions → nindexer.reactions
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_reactions_fid_target_hash
   ON nindexer.reactions (fid, target_hash);

CREATE INDEX CONCURRENTLY idx_reactions_target_hash_fid
  ON nindexer.reactions (target_hash, fid);

-- 4. NEEDS UPDATE: farcaster.casts → nindexer.casts
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_casts_created_at
  ON nindexer.casts (created_at);

-- 5. NO CHANGE: user_labels stays in farcaster schema
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_labels_target_fid_label_value
   ON farcaster.user_labels (target_fid, label_value);

-- 6. NEEDS UPDATE: farcaster.casts → nindexer.casts
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_casts_fid
   ON nindexer.casts (fid);

-- 7. NO CHANGE: user_labels stays in farcaster schema
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_labels_lbl2
  ON farcaster.user_labels (target_fid)
  WHERE label_value = '2';