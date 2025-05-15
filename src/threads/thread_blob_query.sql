BEGIN;
SET LOCAL jit = off;

WITH
-------------------------------------------------------------------------------
-- 0) Root cast's hash
-------------------------------------------------------------------------------
root_hash AS (
  SELECT DECODE('46e34ad768809ea0bc12c2ed90fdcfa1c22bae2f', 'hex') AS val
),

-------------------------------------------------------------------------------
-- 1) Identify the OP's fid by looking at the root cast
-------------------------------------------------------------------------------
op_fid AS (
  SELECT c.fid AS opfid
  FROM farcaster.casts c
  JOIN root_hash rh ON c.hash = rh.val
  LIMIT 1
),

-------------------------------------------------------------------------------
-- 2) Recursively build the full thread from that root, depth <= 10
-------------------------------------------------------------------------------
base_thread AS (
  WITH RECURSIVE all_casts AS (
    SELECT
      c.hash,
      c.parent_hash,
      c.text,
      p.username,
      c.embeds,
      c.fid,
      0 AS depth,
      ARRAY[c.hash] AS path
    FROM farcaster.casts c
    JOIN nindexer.profiles p ON c.fid = p.fid
    JOIN root_hash rh ON c.hash = rh.val  -- root = depth 0

    UNION ALL

    SELECT
      child.hash,
      child.parent_hash,
      child.text,
      cp.username,
      child.embeds,
      child.fid,
      parent.depth + 1 AS depth,
      parent.path || child.hash AS path
    FROM farcaster.casts child
    JOIN nindexer.profiles cp ON child.fid = cp.fid
    JOIN all_casts parent ON child.parent_hash = parent.hash
    WHERE parent.depth < 10
  )
  SELECT * FROM all_casts
),

-------------------------------------------------------------------------------
-- 3) Gather reaction counts for the entire subtree
-------------------------------------------------------------------------------
extended_rc AS (
  SELECT
    bt.hash,
    COALESCE(rc.like_count, 0)    AS like_count,
    COALESCE(rc.recast_count, 0)  AS recast_count,
    COALESCE(rc.comment_count, 0) AS comment_count
  FROM base_thread bt
  LEFT JOIN unbias.reaction_counts rc ON rc.hash = bt.hash
),

-------------------------------------------------------------------------------
-- 4) OP nodes: authored or personally reacted
-------------------------------------------------------------------------------
op_nodes AS (
  SELECT DISTINCT bt.hash
  FROM base_thread bt
  JOIN op_fid ON bt.fid = op_fid.opfid  -- OP is author

  UNION

  SELECT DISTINCT bt.hash
  FROM base_thread bt
  JOIN farcaster.reactions r ON r.target_hash = bt.hash
  JOIN op_fid ON r.fid = op_fid.opfid   -- OP personally reacted
),

-------------------------------------------------------------------------------
-- 5) Climb upward from each op_node to the root
-------------------------------------------------------------------------------
climb_op AS (
  WITH RECURSIVE cte_climb AS (
    SELECT n.hash, bt.parent_hash
    FROM op_nodes n
    JOIN base_thread bt ON bt.hash = n.hash

    UNION ALL

    SELECT parent.hash, parent.parent_hash
    FROM cte_climb c
    JOIN base_thread parent ON c.parent_hash = parent.hash
  )
  SELECT DISTINCT hash FROM cte_climb
),

-------------------------------------------------------------------------------
-- 6) needed_nodes = union of op_nodes + their ancestors
-------------------------------------------------------------------------------
needed_nodes AS (
  SELECT hash FROM op_nodes
  UNION
  SELECT hash FROM climb_op
),

-------------------------------------------------------------------------------
-- 7) single most popular cast (excluding root)
-------------------------------------------------------------------------------
top_cast AS (
  SELECT hash
  FROM (
    SELECT
      er.hash,
      (er.like_count + er.recast_count + er.comment_count) AS popularity
    FROM extended_rc er
    JOIN base_thread bt ON bt.hash = er.hash
    JOIN root_hash rh ON bt.hash <> rh.val    -- exclude root
    WHERE bt.parent_hash IS NOT NULL          -- must be a comment
    ORDER BY popularity DESC
    LIMIT 1
  ) sub
),

-------------------------------------------------------------------------------
-- 8) star_fid: the fid of the user who authored that top_cast
-------------------------------------------------------------------------------
star_fid AS (
  SELECT c.fid AS star_user_fid
  FROM top_cast tc
  JOIN farcaster.casts c ON c.hash = tc.hash
  LIMIT 1
),

-------------------------------------------------------------------------------
-- 9) star_nodes: 
--    1) All casts in base_thread that "star user" either authored OR reacted to.
--    2) Exclude root, must be a comment.
--    3) Sort so authored=first, then popularity in each group.
--    4) Limit top 10.
-------------------------------------------------------------------------------
star_nodes AS (
  SELECT sn.hash
  FROM (
    SELECT
      bt.hash,
      CASE WHEN bt.fid = sf.star_user_fid THEN 1 ELSE 0 END AS is_authored,
      (er.like_count + er.recast_count + er.comment_count) AS popularity
    FROM base_thread bt
    JOIN star_fid sf
      ON TRUE  -- just to attach star_user_fid
    LEFT JOIN farcaster.reactions r 
      ON r.target_hash = bt.hash
      AND r.fid = sf.star_user_fid
    LEFT JOIN extended_rc er 
      ON er.hash = bt.hash
    JOIN root_hash rh 
      ON bt.hash <> rh.val
    WHERE bt.parent_hash IS NOT NULL
      AND (bt.fid = sf.star_user_fid OR r.fid = sf.star_user_fid)
    GROUP BY bt.hash, bt.fid, er.like_count, er.recast_count, er.comment_count, sf.star_user_fid
    ORDER BY is_authored DESC, popularity DESC
    LIMIT 10
  ) sn
),

-------------------------------------------------------------------------------
-- 10) climb_star: climb from each star_node up to the root
-------------------------------------------------------------------------------
climb_star AS (
  WITH RECURSIVE cte_climb AS (
    SELECT s.hash, bt.parent_hash
    FROM star_nodes s
    JOIN base_thread bt ON bt.hash = s.hash

    UNION ALL

    SELECT parent.hash, parent.parent_hash
    FROM cte_climb c
    JOIN base_thread parent ON c.parent_hash = parent.hash
  )
  SELECT DISTINCT hash FROM cte_climb
),

-------------------------------------------------------------------------------
-- 11) star_needed = star_nodes + their ancestors
-------------------------------------------------------------------------------
star_needed AS (
  SELECT hash FROM star_nodes
  UNION
  SELECT hash FROM climb_star
),

-------------------------------------------------------------------------------
-- 12) extended_needed = needed_nodes + top_cast + star_needed
-------------------------------------------------------------------------------
extended_needed AS (
  SELECT hash FROM needed_nodes
  UNION
  SELECT hash FROM top_cast
  UNION
  SELECT hash FROM star_needed
),

-------------------------------------------------------------------------------
-- 12.5) forced_in: any cast that is in extended_needed but NOT in needed_nodes
-------------------------------------------------------------------------------
forced_in AS (
  SELECT hash
  FROM extended_needed
  EXCEPT
  SELECT hash FROM needed_nodes
),

-------------------------------------------------------------------------------
-- 13) prune base_thread => only extended_needed
-------------------------------------------------------------------------------
pruned_base AS (
  SELECT
    bt.hash,
    bt.parent_hash,
    bt.text,
    bt.username,
    bt.embeds,
    bt.fid,
    bt.depth,
    bt.path
  FROM base_thread bt
  JOIN extended_needed en ON bt.hash = en.hash
),

-------------------------------------------------------------------------------
-- 14) Parse single embed->0
-------------------------------------------------------------------------------
parsed_embeds AS (
  SELECT
    pb.hash             AS cast_hash,
    pb.text             AS cast_text,
    pb.username         AS cast_author,
    pb.depth,
    pb.path,
    DECODE(
      STRING_AGG(LPAD(TO_HEX(elem::int), 2, '0'), ''),
      'hex'
    ) AS embed_hash_bytea
  FROM pruned_base pb
  CROSS JOIN LATERAL (
    SELECT elem
    FROM JSON_ARRAY_ELEMENTS_TEXT(pb.embeds::json->0->'castId'->'hash'->'data') AS elem
  ) embed
  GROUP BY pb.hash, pb.text, pb.username, pb.depth, pb.path
),

-------------------------------------------------------------------------------
-- 15) Join parsed embed => quoted cast
-------------------------------------------------------------------------------
joined_quotes AS (
  SELECT
    pe.cast_hash,
    pe.cast_text,
    pe.cast_author,
    pe.depth,
    pe.path,
    orig.text     AS original_text,
    prof.username AS original_author
  FROM parsed_embeds pe
  LEFT JOIN farcaster.casts orig
         ON orig.hash = pe.embed_hash_bytea
  LEFT JOIN nindexer.profiles prof
         ON orig.fid = prof.fid
),

-------------------------------------------------------------------------------
-- 16) Filter extended_rc => only pruned_base
-------------------------------------------------------------------------------
filtered_rc AS (
  SELECT
    pb.hash,
    er.like_count,
    er.recast_count,
    er.comment_count
  FROM pruned_base pb
  JOIN extended_rc er ON pb.hash = er.hash
),

-------------------------------------------------------------------------------
-- 17) Combine everything
-------------------------------------------------------------------------------
final_thread AS (
  SELECT
    pb.hash,
    pb.parent_hash,
    pb.fid AS fid,  -- ensure we expose the fid for final output
    pb.depth,
    pb.path,
    jq.original_text,
    jq.original_author,
    fr.like_count,
    fr.recast_count,
    fr.comment_count,
    pb.username,
    pb.text,

    CASE WHEN tc.hash IS NOT NULL THEN TRUE ELSE FALSE END AS is_top_popular,
    CASE WHEN sn.hash IS NOT NULL THEN TRUE ELSE FALSE END AS is_star_node,
    CASE WHEN fi.hash IS NOT NULL THEN TRUE ELSE FALSE END AS was_forced_in
  FROM pruned_base pb
  LEFT JOIN joined_quotes jq ON pb.hash = jq.cast_hash
  LEFT JOIN filtered_rc  fr ON pb.hash = fr.hash
  LEFT JOIN top_cast     tc ON pb.hash = tc.hash
  LEFT JOIN star_nodes   sn ON pb.hash = sn.hash
  LEFT JOIN forced_in    fi ON pb.hash = fi.hash
)

-------------------------------------------------------------------------------
-- 18) Output final
-------------------------------------------------------------------------------
SELECT
  -----------------------------------------------------------------------------
  -- 1) The aggregated conversation text, no mention of stars or counts
  -----------------------------------------------------------------------------
  STRING_AGG(
    CONCAT(
      REPEAT('-', ft.depth),
      '👤@', ft.username,
      ': ',
      COALESCE(
        REGEXP_REPLACE(
          ft.text,
          'https:\/\/warpcast\.com\/[^ ]+\/0x[0-9A-Fa-f]{8,}',
          'QUOTE:["' || ft.original_text || '" - @' || ft.original_author || ']',
          'g'
        ),
        ft.text
      )
    ),
    E'\n'
    ORDER BY ft.path
  ) AS thread_blob,

  -----------------------------------------------------------------------------
  -- 2) Distinct array of fids, sorted by descending popularity (like+recast+comment)
  -----------------------------------------------------------------------------
  (
    SELECT ARRAY_AGG(sub.fid ORDER BY sub.popularity DESC)
    FROM (
      -- group each fid by max popularity so each fid appears only once
      SELECT 
        f2.fid,
        MAX(f2.like_count + f2.recast_count + f2.comment_count) AS popularity
      FROM final_thread f2
      WHERE f2.fid IS NOT NULL
      GROUP BY f2.fid
    ) sub
  ) AS fids_array

FROM final_thread ft;

COMMIT;
