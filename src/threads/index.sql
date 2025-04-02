CREATE INDEX CONCURRENTLY idx_casts_root_parent_hash
    ON farcaster.casts (root_parent_hash);

CREATE INDEX CONCURRENTLY idx_casts_parent_hash
    ON farcaster.casts (parent_hash);

