CREATE MATERIALIZED VIEW unbias.reaction_counts
TABLESPACE pg_default
AS 
WITH reaction_counts AS (
    SELECT 
        r_1.target_hash,
        count(*) FILTER (WHERE r_1.reaction_type = 1) AS like_count,
        count(*) FILTER (WHERE r_1.reaction_type = 2) AS recast_count
    FROM farcaster.reactions r_1
    JOIN farcaster.user_labels ul ON r_1.fid = ul.target_fid
    WHERE ul.label_value::integer = 2
    GROUP BY r_1.target_hash
), comment_counts AS (
    SELECT 
        c_1.parent_hash AS target_hash,
        count(*) AS comment_count
    FROM farcaster.casts c_1
    JOIN farcaster.user_labels ul ON c_1.fid = ul.target_fid
    WHERE ul.label_value::integer = 2
    GROUP BY c_1.parent_hash
), combined AS (
    SELECT 
        COALESCE(r.target_hash, c.target_hash) AS target_hash,
        COALESCE(r.like_count, 0::bigint) AS like_count,
        COALESCE(r.recast_count, 0::bigint) AS recast_count,
        COALESCE(c.comment_count, 0::bigint) AS comment_count
    FROM reaction_counts r
    FULL JOIN comment_counts c ON r.target_hash = c.target_hash
)
SELECT 
    combined.target_hash AS hash,
    farcaster.casts.root_parent_hash,
    combined.like_count,
    combined.recast_count,
    combined.comment_count
FROM combined
JOIN farcaster.casts ON combined.target_hash = farcaster.casts.hash
WITH DATA;


-- Make sure the materialized view can meet you halfway
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_reaction_counts_hash
  ON unbias.reaction_counts (hash);