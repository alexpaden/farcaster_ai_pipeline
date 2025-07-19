-- FAST & COMPLETE Thread Query - Includes full quote hydration with optimized profile lookups
-- This version maintains ALL functionality while optimizing performance
-- UPDATED: Filters out replies shorter than configurable character length after quote hydration
-- CONFIGURATION: Edit min_reply_length in the config CTE below (default: 20)

WITH config AS (
    -- CONFIGURATION: Change min_reply_length to filter replies (e.g., 20, 200, 2000)
    -- This filters AFTER quotes/links are hydrated, so char count includes expanded content
    SELECT 20 AS min_reply_length
),
target_thread AS (
    SELECT decode('DC8FD3E4047AD9BFC556EB4ACAAFBB86EE9C4C81','hex') AS hash
),

-- Check thread size
thread_size_check AS (
    SELECT 
        COUNT(*) AS reply_count,
        MAX(CASE WHEN c.hash = tt.hash THEN c.text END) AS root_text,
        MAX(CASE WHEN c.hash = tt.hash THEN c.fid END) AS root_fid
    FROM target_thread tt
    CROSS JOIN nindexer.casts c
    WHERE c.root_parent_hash = tt.hash OR c.hash = tt.hash
),

-- Get root info with username
root_info AS (
    SELECT 
        tsc.*,
        p.username AS root_username
    FROM thread_size_check tsc
    LEFT JOIN nindexer.profiles p ON p.fid = tsc.root_fid
),

-- Get all casts in the thread
all_thread_casts AS MATERIALIZED (
  WITH RECURSIVE thread_builder AS (
    SELECT c.hash, c.parent_hash, c.text, c.embeds, c.fid, 
           c.embedded_urls, c.embedded_casts, c.mentions, c.mentions_positions,
           0 AS depth, ARRAY[c.hash] AS path, c.fid AS op_fid
    FROM nindexer.casts c
    CROSS JOIN target_thread tt
    WHERE c.hash = tt.hash
    
    UNION ALL
    
    SELECT c.hash, c.parent_hash, c.text, c.embeds, c.fid,
           c.embedded_urls, c.embedded_casts, c.mentions, c.mentions_positions,
           p.depth + 1, p.path || c.hash, p.op_fid
    FROM nindexer.casts c
    JOIN thread_builder p ON c.parent_hash = p.hash
    WHERE p.depth < 10
  )
  SELECT * FROM thread_builder
),

-- Reaction counts with EXISTS optimization
thread_reactions AS MATERIALIZED (
  SELECT 
    r.target_hash AS hash,
    COUNT(*) AS reaction_count
  FROM nindexer.reactions r
  WHERE r.target_hash IN (SELECT hash FROM all_thread_casts)
    AND EXISTS (
      SELECT 1 FROM farcaster.user_labels ul 
      WHERE ul.target_fid = r.fid 
      AND ul.label_value::int = 2
      LIMIT 1
    )
  GROUP BY r.target_hash
),

-- Build thread with data
thread_with_data AS MATERIALIZED (
  SELECT 
    atc.*,
    p.username,
    COALESCE(tr.reaction_count, 0) AS reaction_count
  FROM all_thread_casts atc
  JOIN nindexer.profiles p ON p.fid = atc.fid
  LEFT JOIN thread_reactions tr ON tr.hash = atc.hash
),

-- Filter thread
filtered_thread AS MATERIALIZED (
  WITH op_interactions AS (
    SELECT DISTINCT t.hash
    FROM thread_with_data t
    WHERE t.fid = t.op_fid
    
    UNION
    
    SELECT DISTINCT t.hash
    FROM thread_with_data t
    WHERE EXISTS (
      SELECT 1 FROM nindexer.reactions r 
      WHERE r.target_hash = t.hash AND r.fid = t.op_fid
      LIMIT 1
    )
  ),
  popular_extras AS (
    SELECT hash
    FROM (
      SELECT hash, reaction_count,
             ROW_NUMBER() OVER (ORDER BY reaction_count DESC) AS rn,
             COUNT(*) OVER () AS total
      FROM thread_with_data
      WHERE reaction_count > 0
    ) s
    WHERE rn <= GREATEST(2, CEIL(total * 0.05))
  ),
  all_needed AS (
    SELECT DISTINCT ancestor_hash
    FROM (
      SELECT hash FROM op_interactions
      UNION
      SELECT hash FROM popular_extras
    ) x
    JOIN thread_with_data t ON t.hash = x.hash
    CROSS JOIN LATERAL UNNEST(t.path) AS ancestor_hash
  )
  SELECT t.*
  FROM thread_with_data t
  WHERE EXISTS (SELECT 1 FROM all_needed an WHERE t.hash = an.ancestor_hash)
),

-- Calculate subtree popularity
with_popularity AS MATERIALIZED (
  SELECT 
    ft.*,
    (
      SELECT MAX(descendant.reaction_count)
      FROM filtered_thread descendant
      WHERE descendant.path @> ARRAY[ft.hash]
    ) AS max_subtree_popularity
  FROM filtered_thread ft
),

-------------------------------------------------------------------------------
-- OPTIMIZED FULL QUOTE HYDRATION - Maintains all functionality
-------------------------------------------------------------------------------

-- Step 1: Pre-collect ALL FIDs we'll need for the entire quote tree
all_quote_fids AS MATERIALIZED (
  WITH RECURSIVE quote_scan AS (
    -- Start with main thread casts
    SELECT 
      hash,
      fid,
      mentions,
      embedded_casts,
      0 as depth
    FROM with_popularity
    
    UNION ALL
    
    -- Scan embedded casts recursively
    SELECT 
      c.hash,
      c.fid,
      c.mentions,
      c.embedded_casts,
      qs.depth + 1
    FROM quote_scan qs
    CROSS JOIN LATERAL UNNEST(qs.embedded_casts) AS ec(hash)
    JOIN nindexer.casts c ON c.hash = ec.hash
    WHERE qs.depth < 3
  )
  SELECT DISTINCT fid FROM (
    -- All cast authors
    SELECT fid FROM quote_scan
    UNION
    -- All mentioned users
    SELECT UNNEST(mentions) as fid 
    FROM quote_scan 
    WHERE mentions IS NOT NULL AND array_length(mentions, 1) > 0
  ) all_fids
  WHERE fid IS NOT NULL
),

-- Step 2: Load all needed profiles in one go - this is our optimization!
quote_profiles AS MATERIALIZED (
  SELECT fid, username
  FROM nindexer.profiles
  WHERE fid IN (SELECT fid FROM all_quote_fids)
),

-- Step 3: Process quotes
quote_rec AS MATERIALIZED (
  WITH RECURSIVE quote_builder AS (
    SELECT  
      wp.hash, 
      wp.text,
      wp.embedded_urls, 
      wp.embedded_casts,
      wp.mentions, 
      wp.mentions_positions,
      wp.fid, 
      wp.username,
      0 AS quote_level,
      ARRAY[wp.hash]::bytea[] AS quote_path
    FROM with_popularity wp
    
    UNION ALL
    
    SELECT  
      c.hash, 
      c.text,
      c.embedded_urls, 
      c.embedded_casts,
      c.mentions, 
      c.mentions_positions,
      c.fid, 
      qp.username,
      qb.quote_level + 1,
      qb.quote_path || c.hash
    FROM quote_builder qb
    CROSS JOIN LATERAL (
      SELECT DISTINCT h.hash
      FROM unnest(qb.embedded_casts) AS h(hash)
    ) h
    JOIN nindexer.casts c ON c.hash = h.hash
    JOIN quote_profiles qp ON qp.fid = c.fid  -- Use pre-loaded profiles
    WHERE qb.quote_level < 3
      AND array_position(qb.quote_path, c.hash) IS NULL
  )
  SELECT * FROM quote_builder
),

-- Step 4: Process mentions with pre-loaded profiles
quote_pos AS (
  SELECT DISTINCT ON (qr.hash, tag)
    qr.hash,
    length(
      convert_from(
        substring(qr_bytes.utf8_bytes
                  FROM 1 FOR 
                  -- Adjust position if it points to a UTF-8 continuation byte
                  CASE 
                    WHEN qr.mentions_positions[i] = 0 THEN qr.mentions_positions[i]
                    WHEN qr.mentions_positions[i] >= octet_length(qr_bytes.utf8_bytes) THEN octet_length(qr_bytes.utf8_bytes)
                    -- Check if current position is a continuation byte (10xxxxxx = 128-191)
                    WHEN get_byte(qr_bytes.utf8_bytes, qr.mentions_positions[i]) BETWEEN 128 AND 191
                    THEN 
                      -- Find next valid UTF-8 boundary (skip to next character start)
                      CASE
                        WHEN qr.mentions_positions[i]+1 >= octet_length(qr_bytes.utf8_bytes) THEN octet_length(qr_bytes.utf8_bytes)
                        -- Next byte is ASCII (< 128) or multi-byte start (>= 192)
                        WHEN get_byte(qr_bytes.utf8_bytes, qr.mentions_positions[i]+1) < 128 
                          OR get_byte(qr_bytes.utf8_bytes, qr.mentions_positions[i]+1) >= 192 THEN qr.mentions_positions[i] + 1
                        WHEN qr.mentions_positions[i]+2 >= octet_length(qr_bytes.utf8_bytes) THEN octet_length(qr_bytes.utf8_bytes)
                        WHEN get_byte(qr_bytes.utf8_bytes, qr.mentions_positions[i]+2) < 128 
                          OR get_byte(qr_bytes.utf8_bytes, qr.mentions_positions[i]+2) >= 192 THEN qr.mentions_positions[i] + 2
                        WHEN qr.mentions_positions[i]+3 >= octet_length(qr_bytes.utf8_bytes) THEN octet_length(qr_bytes.utf8_bytes)
                        WHEN get_byte(qr_bytes.utf8_bytes, qr.mentions_positions[i]+3) < 128 
                          OR get_byte(qr_bytes.utf8_bytes, qr.mentions_positions[i]+3) >= 192 THEN qr.mentions_positions[i] + 3
                        ELSE qr.mentions_positions[i] + 4
                      END
                    ELSE qr.mentions_positions[i]
                  END),
        'utf8')) AS char_pos,
    tag
  FROM quote_rec qr
  -- Convert text to UTF-8 bytes once per row to avoid repeated conversions
  CROSS JOIN LATERAL (SELECT convert_to(qr.text,'utf8') AS utf8_bytes) qr_bytes
  CROSS JOIN LATERAL generate_subscripts(qr.mentions,1) AS gs(i)
  JOIN quote_profiles qp ON qp.fid = qr.mentions[gs.i]  -- Use pre-loaded profiles
  CROSS JOIN LATERAL (SELECT '@'||qp.username AS tag) t
  WHERE qr.mentions IS NOT NULL 
    AND array_length(qr.mentions, 1) > 0
  ORDER BY qr.hash, tag, char_pos
),

quote_seg AS (
  SELECT  
    qp.hash,
    qp.char_pos,
    substr(qr.text,
           COALESCE(lag(qp.char_pos) OVER w,0)+1,
           qp.char_pos-COALESCE(lag(qp.char_pos) OVER w,0))
    || qp.tag AS chunk
  FROM quote_pos qp
  JOIN quote_rec qr USING (hash)
  WINDOW w AS (PARTITION BY qp.hash ORDER BY qp.char_pos)
),

quote_rebuild AS (
  SELECT 
    hash,
    string_agg(chunk,'' ORDER BY char_pos) AS head,
    max(char_pos) AS last_char
  FROM quote_seg
  GROUP BY hash
),

-- Handle missing URLs
quote_missing_urls AS (
  SELECT  
    qr.hash,
    array_agg(DISTINCT url) AS urls
  FROM quote_rec qr
  CROSS JOIN unnest(qr.embedded_urls) AS u(url)
  WHERE position(lower(url) IN lower(qr.text)) = 0
    AND qr.embedded_urls IS NOT NULL 
    AND array_length(qr.embedded_urls, 1) > 0
  GROUP BY qr.hash
),

-- Build core text with mentions and URLs
quote_core AS (
  SELECT  
    qr.hash,
    COALESCE(qrb.head || substr(qr.text,qrb.last_char+1), qr.text) AS body,
    COALESCE(
      CASE WHEN qmu.urls IS NULL OR array_length(qmu.urls,1)=0
           THEN '' ELSE ' '||array_to_string(qmu.urls,' ') END,'') AS tail,
    qr.username AS author,
    qr.embedded_casts
  FROM quote_rec qr
  LEFT JOIN quote_rebuild qrb USING (hash)
  LEFT JOIN quote_missing_urls qmu USING (hash)
),

-- Build quote tree bottom-up
quote_tree AS (
  WITH RECURSIVE quote_tree_builder AS (
    -- Leaves (no embedded casts)
    SELECT  
      qc.hash,
      qc.body || qc.tail AS full_text,
      qc.author
    FROM quote_core qc
    WHERE qc.embedded_casts IS NULL
       OR array_length(qc.embedded_casts,1)=0

    UNION ALL
    
    -- Parents (with embedded casts)
    SELECT  
      qc.hash,
      qc.body
      || ' ' || format(
             'QUOTE:["%s" - @%s]',
             qtb.full_text,
             qtb.author)
      || qc.tail AS full_text,
      qc.author
    FROM quote_core qc
    CROSS JOIN LATERAL (
      SELECT DISTINCT h.hash
      FROM unnest(qc.embedded_casts) AS h(hash)
    ) h
    JOIN quote_tree_builder qtb ON qtb.hash = h.hash
  )
  SELECT * FROM quote_tree_builder
),

-- Final hydrated text with deduplication, URL removal, and line break normalization
final_quote_text AS MATERIALIZED (
  SELECT DISTINCT ON (hash)
    hash,
    regexp_replace(
      regexp_replace(
        regexp_replace(full_text,
           '(@[A-Za-z0-9._]+)\1+', '\1', 'g'),
        'https://(farcaster\.xyz|warpcast\.com)/[^/]+/0x[a-fA-F0-9]+', '', 'g'),
      E'[\\n\\r\\t]+', ' ', 'g'
    ) AS hydrated_text
  FROM quote_tree
  ORDER BY hash
),

-- Build sorted thread with proper parent-child nesting
sorted_thread AS MATERIALIZED (
  WITH sibling_order AS (
    -- First, assign sibling order based on popularity
    SELECT 
      wp.*,
      ROW_NUMBER() OVER (
        PARTITION BY COALESCE(wp.parent_hash, '\x00'::bytea) 
        ORDER BY wp.max_subtree_popularity DESC NULLS LAST, wp.hash
      ) AS sibling_rank,
      COALESCE(fqt.hydrated_text, wp.text) AS final_text
    FROM with_popularity wp
    LEFT JOIN final_quote_text fqt ON fqt.hash = wp.hash
  ),
  ordered_build AS (
    -- Build paths using recursive CTE
    WITH RECURSIVE path_builder AS (
      -- Root
      SELECT
        so.*,
        ARRAY[LPAD(so.sibling_rank::text, 9, '0')] AS sort_path
      FROM sibling_order so
      WHERE so.parent_hash IS NULL
      
      UNION ALL
      
      -- Children - append sibling rank to parent's path
      SELECT
        child.*,
        parent.sort_path || LPAD(child.sibling_rank::text, 9, '0')
      FROM sibling_order child
      JOIN path_builder parent ON child.parent_hash = parent.hash
    )
    SELECT * FROM path_builder
  )
  SELECT * FROM ordered_build
)

-- Final output
SELECT 
    -- 1. THREAD BLOB
    CASE 
        WHEN ri.reply_count > 50000 THEN
            CONCAT('@', ri.root_username, ': ', ri.root_text, E'\n\n(thread too large: ', ri.reply_count, ' replies)')
        ELSE
            (
                SELECT STRING_AGG(
                    CONCAT(
                      CASE WHEN st.depth = 0 THEN '' ELSE '~' || st.depth || ': ' END,
                      '@', st.username,
                      ': ',
                      st.final_text
                    ),
                    E'\n'
                    ORDER BY st.sort_path
                )
                FROM sorted_thread st
                -- FILTER: Exclude replies shorter than configured minimum length after hydration
                CROSS JOIN config
                WHERE st.depth = 0 OR LENGTH(st.final_text) >= config.min_reply_length
            )
    END AS thread_blob,
    
    -- 2. FIDS ARRAY
    CASE 
        WHEN ri.reply_count > 50000 THEN
            ARRAY[ri.root_fid]
        ELSE
            (
                SELECT ARRAY_AGG(sub.fid ORDER BY sub.popularity DESC)
                FROM (
                  SELECT 
                    st2.fid,
                    MAX(st2.reaction_count) AS popularity
                  FROM sorted_thread st2
                  -- FILTER: Only include FIDs from posts that pass the length filter
                  CROSS JOIN config
                  WHERE st2.fid IS NOT NULL
                    AND (st2.depth = 0 OR LENGTH(st2.final_text) >= config.min_reply_length)
                  GROUP BY st2.fid
                ) sub
            )
    END AS fids_array,
    
    -- 3. TOTAL REACTIONS
    CASE 
        WHEN ri.reply_count > 50000 THEN
            (
                SELECT COUNT(*)
                FROM nindexer.reactions r
                WHERE r.target_hash = (SELECT hash FROM target_thread)
                  AND EXISTS (
                    SELECT 1 FROM farcaster.user_labels ul 
                    WHERE ul.target_fid = r.fid 
                    AND ul.label_value::int = 2
                    LIMIT 1
                  )
            )
        ELSE
            (
                SELECT SUM(tr.reaction_count) 
                FROM thread_reactions tr
                -- Join to get depth and final_text for filtering
                JOIN sorted_thread st ON st.hash = tr.hash
                CROSS JOIN config
                WHERE st.depth = 0 OR LENGTH(st.final_text) >= config.min_reply_length
            )
    END AS total_reactions,
    
    -- 4. TOKEN COUNT
    tiktoken_count('cl100k_base', 
        CASE 
            WHEN ri.reply_count > 50000 THEN
                CONCAT('@', ri.root_username, ': ', ri.root_text, E'\n\n(thread too large: ', ri.reply_count, ' replies)')
            ELSE
                (
                    SELECT STRING_AGG(
                        CONCAT(
                          CASE WHEN st.depth = 0 THEN '' ELSE '~' || st.depth || ': ' END,
                          '@', st.username,
                          ': ',
                          st.final_text
                        ),
                        E'\n'
                        ORDER BY st.sort_path
                    )
                    FROM sorted_thread st
                    -- FILTER: Exclude replies shorter than configured minimum length after hydration
                    CROSS JOIN config
                    WHERE st.depth = 0 OR LENGTH(st.final_text) >= config.min_reply_length
                )
        END
    ) AS tokens
FROM root_info ri;