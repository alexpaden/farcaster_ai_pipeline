-- Optimized thread processing query
-- Key optimizations:
-- 1. Combine data fetching with thread building
-- 2. Simplify subtree popularity calculation
-- 3. Inline quote parsing to avoid extra joins
-- 4. Pre-calculate all reaction counts to avoid repeated good_fids lookups

BEGIN;
SET LOCAL jit = off;
SET LOCAL work_mem = '256MB';
WITH
-------------------------------------------------------------------------------
-- 0) Root cast's hash
-------------------------------------------------------------------------------
root_hash AS (
  SELECT DECODE('46e34ad768809ea0bc12c2ed90fdcfa1c22bae2f', 'hex') AS val
),

-------------------------------------------------------------------------------
-- 1) First, get all casts in the thread
-------------------------------------------------------------------------------
all_thread_casts AS (
  WITH RECURSIVE thread_builder AS (
    -- Root
    SELECT c.hash, c.parent_hash, c.text, c.embeds, c.fid, 
           0 AS depth, ARRAY[c.hash] AS path, c.fid AS op_fid
    FROM farcaster.casts c
    JOIN root_hash rh ON c.hash = rh.val
    
    UNION ALL
    
    -- Children
    SELECT c.hash, c.parent_hash, c.text, c.embeds, c.fid,
           p.depth + 1, p.path || c.hash, p.op_fid
    FROM farcaster.casts c
    JOIN thread_builder p ON c.parent_hash = p.hash
    WHERE p.depth < 10
  )
  SELECT * FROM thread_builder
),

-------------------------------------------------------------------------------
-- 2) Pre-calculate reaction counts for all thread casts
-------------------------------------------------------------------------------
thread_reactions AS (
  SELECT 
    r.target_hash AS hash,
    COUNT(*) AS reaction_count
  FROM farcaster.reactions r
  JOIN farcaster.user_labels ul ON ul.target_fid = r.fid AND ul.label_value::int = 2
  WHERE r.target_hash IN (SELECT hash FROM all_thread_casts)
  GROUP BY r.target_hash
),

-------------------------------------------------------------------------------
-- 3) Build thread with all data including reaction counts
-------------------------------------------------------------------------------
thread_with_data AS (
  SELECT 
    atc.*,
    p.username,
    COALESCE(tr.reaction_count, 0) AS reaction_count
  FROM all_thread_casts atc
  JOIN nindexer.profiles p ON p.fid = atc.fid
  LEFT JOIN thread_reactions tr ON tr.hash = atc.hash
),

-------------------------------------------------------------------------------
-- 4) Get OP interactions and filter nodes
-------------------------------------------------------------------------------
filtered_thread AS (
  -- Mark nodes that OP interacted with
  WITH op_interactions AS (
    SELECT DISTINCT t.hash
    FROM thread_with_data t
    WHERE t.fid = t.op_fid -- OP authored
    
    UNION
    
    SELECT DISTINCT t.hash
    FROM thread_with_data t
    JOIN farcaster.reactions r ON r.target_hash = t.hash AND r.fid = t.op_fid
  ),
  -- Get all ancestors of needed nodes using path arrays
  all_needed AS (
    SELECT DISTINCT ancestor_hash
    FROM op_interactions oi
    JOIN thread_with_data t ON t.hash = oi.hash
    CROSS JOIN LATERAL UNNEST(t.path) AS ancestor_hash
  )
  SELECT t.*
  FROM thread_with_data t
  JOIN all_needed an ON t.hash = an.ancestor_hash
),

-------------------------------------------------------------------------------
-- 5) Calculate subtree popularity
-------------------------------------------------------------------------------
with_popularity AS (
  SELECT 
    ft.*,
    -- For each node, find max popularity in its subtree
    (
      SELECT MAX(descendant.reaction_count)
      FROM filtered_thread descendant
      WHERE descendant.path @> ARRAY[ft.hash]
    ) AS max_subtree_popularity
  FROM filtered_thread ft
),

-------------------------------------------------------------------------------
-- 6) Build sorted thread with inline quote parsing
-------------------------------------------------------------------------------
sorted_thread AS (
  WITH RECURSIVE ordered_build AS (
    -- Root
    SELECT
      wp.*,
      ARRAY[LPAD(TO_CHAR(999999999 - COALESCE(wp.max_subtree_popularity, 0), 'FM000000000'), 9, '0')] AS sort_path,
      -- Parse quotes inline - only if embeds exist and have the right structure
      CASE 
        WHEN wp.embeds IS NOT NULL 
         AND wp.embeds::text != '[]' 
         AND wp.embeds::text LIKE '%castId%' THEN
          (
            WITH embed_data AS (
              SELECT DECODE(
                STRING_AGG(LPAD(TO_HEX(elem::int), 2, '0'), ''), 'hex'
              ) AS embed_hash
              FROM JSON_ARRAY_ELEMENTS_TEXT(wp.embeds::json->0->'castId'->'hash'->'data') AS elem
            )
            SELECT CONCAT('QUOTE:["', c.text, '" - @', p.username, ']')
            FROM embed_data ed
            LEFT JOIN farcaster.casts c ON c.hash = ed.embed_hash
            LEFT JOIN nindexer.profiles p ON c.fid = p.fid
            LIMIT 1
          )
        ELSE NULL
      END AS quote_text
    FROM with_popularity wp
    WHERE wp.parent_hash IS NULL
    
    UNION ALL
    
    -- Children
    SELECT
      child.*,
      parent.sort_path || LPAD(TO_CHAR(999999999 - COALESCE(child.max_subtree_popularity, 0), 'FM000000000'), 9, '0'),
      CASE 
        WHEN child.embeds IS NOT NULL 
         AND child.embeds::text != '[]' 
         AND child.embeds::text LIKE '%castId%' THEN
          (
            WITH embed_data AS (
              SELECT DECODE(
                STRING_AGG(LPAD(TO_HEX(elem::int), 2, '0'), ''), 'hex'
              ) AS embed_hash
              FROM JSON_ARRAY_ELEMENTS_TEXT(child.embeds::json->0->'castId'->'hash'->'data') AS elem
            )
            SELECT CONCAT('QUOTE:["', c.text, '" - @', p.username, ']')
            FROM embed_data ed
            LEFT JOIN farcaster.casts c ON c.hash = ed.embed_hash
            LEFT JOIN nindexer.profiles p ON c.fid = p.fid
            LIMIT 1
          )
        ELSE NULL
      END
    FROM with_popularity child
    JOIN ordered_build parent ON child.parent_hash = parent.hash
  )
  SELECT * FROM ordered_build
)

-------------------------------------------------------------------------------
-- 7) Final output
-------------------------------------------------------------------------------
SELECT
  -- Thread text
  STRING_AGG(
    CONCAT(
      REPEAT('↳', st.depth),
      '@', st.username,
      ': ',
      CASE 
        WHEN st.quote_text IS NOT NULL THEN
          REGEXP_REPLACE(st.text, 'https:\/\/warpcast\.com\/[^ ]+\/0x[0-9A-Fa-f]{8,}', st.quote_text, 'g')
        ELSE st.text
      END
    ),
    E'\n'
    ORDER BY st.sort_path
  ) AS thread_blob,
  
  -- FIDs array ordered by popularity
  (
    SELECT ARRAY_AGG(sub.fid ORDER BY sub.popularity DESC)
    FROM (
      SELECT 
        st2.fid,
        MAX(st2.reaction_count) AS popularity
      FROM sorted_thread st2
      WHERE st2.fid IS NOT NULL
      GROUP BY st2.fid
    ) sub
  ) AS fids_array

FROM sorted_thread st;

COMMIT; 