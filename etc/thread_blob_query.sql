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
-- 7) Calculate max popularity for each subtree (for sorting)
-------------------------------------------------------------------------------
subtree_popularity AS (
  SELECT 
    ancestor.hash,
    MAX(COALESCE(desc_er.like_count + desc_er.recast_count + desc_er.comment_count, 0)) AS max_subtree_popularity
  FROM base_thread ancestor
  JOIN base_thread descendant ON descendant.path @> ARRAY[ancestor.hash]
  LEFT JOIN extended_rc desc_er ON desc_er.hash = descendant.hash
  GROUP BY ancestor.hash
),

-------------------------------------------------------------------------------
-- 8) prune base_thread => only needed_nodes
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
    bt.path,
    sp.max_subtree_popularity
  FROM base_thread bt
  JOIN needed_nodes nn ON bt.hash = nn.hash
  LEFT JOIN subtree_popularity sp ON bt.hash = sp.hash
),

-------------------------------------------------------------------------------
-- 9) Build new paths based on popularity ordering
-------------------------------------------------------------------------------
ordered_thread AS (
  WITH RECURSIVE ordered_build AS (
    -- Start with root
    SELECT
      pb.hash,
      pb.parent_hash,
      pb.text,
      pb.username,
      pb.embeds,
      pb.fid,
      pb.depth,
      ARRAY[pb.hash] AS path,
      pb.max_subtree_popularity,
      ARRAY[LPAD(TO_CHAR(999999999 - COALESCE(pb.max_subtree_popularity, 0), 'FM000000000'), 9, '0')] AS sort_path
    FROM pruned_base pb
    JOIN root_hash rh ON pb.hash = rh.val
    
    UNION ALL
    
    -- Add children, ordered by popularity
    SELECT
      child.hash,
      child.parent_hash,
      child.text,
      child.username,
      child.embeds,
      child.fid,
      child.depth,
      parent.path || child.hash AS path,
      child.max_subtree_popularity,
      parent.sort_path || LPAD(TO_CHAR(999999999 - COALESCE(child.max_subtree_popularity, 0), 'FM000000000'), 9, '0') AS sort_path
    FROM pruned_base child
    JOIN ordered_build parent ON child.parent_hash = parent.hash
  )
  SELECT * FROM ordered_build
),

-------------------------------------------------------------------------------
-- 10) Parse single embed->0
-------------------------------------------------------------------------------
parsed_embeds AS (
  SELECT
    ot.hash             AS cast_hash,
    ot.text             AS cast_text,
    ot.username         AS cast_author,
    ot.depth,
    ot.sort_path,
    DECODE(
      STRING_AGG(LPAD(TO_HEX(elem::int), 2, '0'), ''),
      'hex'
    ) AS embed_hash_bytea
  FROM ordered_thread ot
  CROSS JOIN LATERAL (
    SELECT elem
    FROM JSON_ARRAY_ELEMENTS_TEXT(ot.embeds::json->0->'castId'->'hash'->'data') AS elem
  ) embed
  GROUP BY ot.hash, ot.text, ot.username, ot.depth, ot.sort_path
),

-------------------------------------------------------------------------------
-- 11) Join parsed embed => quoted cast
-------------------------------------------------------------------------------
joined_quotes AS (
  SELECT
    pe.cast_hash,
    pe.cast_text,
    pe.cast_author,
    pe.depth,
    pe.sort_path,
    orig.text     AS original_text,
    prof.username AS original_author
  FROM parsed_embeds pe
  LEFT JOIN farcaster.casts orig
         ON orig.hash = pe.embed_hash_bytea
  LEFT JOIN nindexer.profiles prof
         ON orig.fid = prof.fid
),

-------------------------------------------------------------------------------
-- 12) Filter extended_rc => only ordered_thread
-------------------------------------------------------------------------------
filtered_rc AS (
  SELECT
    ot.hash,
    er.like_count,
    er.recast_count,
    er.comment_count
  FROM ordered_thread ot
  JOIN extended_rc er ON ot.hash = er.hash
),

-------------------------------------------------------------------------------
-- 13) Combine everything
-------------------------------------------------------------------------------
final_thread AS (
  SELECT
    ot.hash,
    ot.parent_hash,
    ot.fid AS fid,
    ot.depth,
    ot.sort_path,
    jq.original_text,
    jq.original_author,
    fr.like_count,
    fr.recast_count,
    fr.comment_count,
    ot.username,
    ot.text
  FROM ordered_thread ot
  LEFT JOIN joined_quotes jq ON ot.hash = jq.cast_hash
  LEFT JOIN filtered_rc  fr ON ot.hash = fr.hash
)

-------------------------------------------------------------------------------
-- 14) Output final
-------------------------------------------------------------------------------
SELECT
  -----------------------------------------------------------------------------
  -- 1) The aggregated conversation text, ordered by popularity
  -----------------------------------------------------------------------------
  STRING_AGG(
    CONCAT(
      REPEAT('↳', ft.depth),
      '@', ft.username,
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
    ORDER BY ft.sort_path
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
