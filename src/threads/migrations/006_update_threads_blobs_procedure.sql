CREATE OR REPLACE PROCEDURE unbias.process_thread_blobs_batch(
    p_batch_size   int DEFAULT 10,
    p_max_batches  int DEFAULT NULL          -- NULL = run to completion
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_rows      int;           -- rows updated in this batch
    v_batches   int := 0;      -- how many batches we've finished
    v_start     timestamptz;   -- batch start time
    v_elapsed   numeric;       -- elapsed seconds (rounded)
    v_first_timestamp timestamptz; -- timestamp of first thread in batch
BEGIN
    LOOP
        ------------------------------------------------------------------
        -- 1. Start processing a batch of threads
        ------------------------------------------------------------------
        v_start := clock_timestamp();
        
        -- Set these for each iteration since COMMIT resets transaction-local settings
        SET LOCAL jit = off;
        SET LOCAL synchronous_commit = off;
        SET LOCAL work_mem = '256MB';

        -- Capture first timestamp for logging
        SELECT timestamp INTO v_first_timestamp
        FROM unbias.threads 
        WHERE threads_status = 0 AND spam = 2 
        ORDER BY timestamp 
        LIMIT 1;

        WITH 
        -- Get batch of unprocessed threads
        batch_threads AS (
            SELECT hash
            FROM unbias.threads
            WHERE threads_status = 0
              AND spam = 2  -- not spam
            ORDER BY timestamp  -- Use the existing index for deterministic ordering
            LIMIT p_batch_size
            FOR UPDATE SKIP LOCKED
        ),
        
        -- Process all threads in the batch
        processed_threads AS (
            SELECT 
                bt.hash,
                thread_data.thread_blob,
                thread_data.fids_array,
                thread_data.total_reactions
            FROM batch_threads bt
            CROSS JOIN LATERAL (
                -- First, check thread size
                WITH thread_size_check AS (
                    SELECT 
                        COUNT(*) AS reply_count,
                        MAX(CASE WHEN c.hash = bt.hash THEN c.text END) AS root_text,
                        MAX(CASE WHEN c.hash = bt.hash THEN c.fid END) AS root_fid
                    FROM farcaster.casts c
                    WHERE c.root_parent_hash = bt.hash OR c.hash = bt.hash
                ),
                -- Get root info with username
                root_info AS (
                    SELECT 
                        tsc.*,
                        p.username AS root_username
                    FROM thread_size_check tsc
                    LEFT JOIN nindexer.profiles p ON p.fid = tsc.root_fid
                )
                SELECT 
                    -- Branch based on thread size
                    CASE 
                        WHEN ri.reply_count > 50000 THEN
                            -- For large threads, just return the original post with a note
                            CONCAT('@', ri.root_username, ': ', ri.root_text, E'\n\n(thread too large: ', ri.reply_count, ' replies)')
                        ELSE
                            -- For normal threads, use the existing complex logic
                            (
                WITH
                -------------------------------------------------------------------------------
                -- 1) First, get all casts in the thread
                -------------------------------------------------------------------------------
                all_thread_casts AS (
                  WITH RECURSIVE thread_builder AS (
                    -- Root
                    SELECT c.hash, c.parent_hash, c.text, c.embeds, c.fid, 
                           0 AS depth, ARRAY[c.hash] AS path, c.fid AS op_fid
                    FROM farcaster.casts c
                    WHERE c.hash = bt.hash
                    
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
                  -- Get top 5% most popular posts (minimum 1)
                  popular_extras AS (
                    SELECT hash
                    FROM (
                      SELECT
                        hash,
                        reaction_count,
                        ROW_NUMBER() OVER (ORDER BY reaction_count DESC) AS rn,
                        COUNT(*) OVER () AS total
                      FROM thread_with_data
                      WHERE reaction_count > 0  -- Only consider posts with reactions
                    ) s
                    WHERE rn <= GREATEST(2, CEIL(total * 0.05))  -- top 5%, but never 0 rows
                  ),
                  -- Get all ancestors of needed nodes using path arrays
                  all_needed AS (
                    SELECT DISTINCT ancestor_hash
                    FROM (
                      SELECT hash FROM op_interactions
                      UNION
                      SELECT hash FROM popular_extras  -- Include popular posts
                    ) x
                    JOIN thread_with_data t ON t.hash = x.hash
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
                      -- Parse quotes inline using regex to handle malformed JSON
                      CASE
                        WHEN wp.embeds::text <> '"[]"' 
                         AND wp.embeds::text LIKE '%castId%'                 -- fast pre-filter
                        THEN (
                          /* 1. Pull the first "…'hash': {'data': [ … ]}" numeric list with a regex
                             2. Split to individual numbers
                             3. int → hex → string_agg → DECODE → bytea  */
                          SELECT CONCAT('QUOTE:["', c.text, '" - @', p.username, ']')
                          FROM  LATERAL (
                                  SELECT DECODE(
                                           STRING_AGG(LPAD(TO_HEX(TRIM(n)::INT), 2, '0'), ''),
                                           'hex'
                                         ) AS embed_hash
                                  FROM   UNNEST(
                                           STRING_TO_ARRAY(
                                             ( REGEXP_MATCH(
                                                 -- Remove outer quotes if present
                                                 TRIM(BOTH '"' FROM wp.embeds::TEXT),
                                                 $re$'hash'\s*:\s*\{'data'\s*:\s*\[([0-9,\s]+)\]$re$
                                               )
                                             )[1]                                         -- the capture group
                                             , ','
                                           )
                                         ) AS n
                                ) h
                          LEFT  JOIN farcaster.casts   c ON c.hash = h.embed_hash
                          LEFT  JOIN nindexer.profiles p ON p.fid  = c.fid
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
                        WHEN child.embeds::text <> '"[]"' 
                         AND child.embeds::text LIKE '%castId%'                 -- fast pre-filter
                        THEN (
                          /* 1. Pull the first "…'hash': {'data': [ … ]}" numeric list with a regex
                             2. Split to individual numbers
                             3. int → hex → string_agg → DECODE → bytea  */
                          SELECT CONCAT('QUOTE:["', c.text, '" - @', p.username, ']')
                          FROM  LATERAL (
                                  SELECT DECODE(
                                           STRING_AGG(LPAD(TO_HEX(TRIM(n)::INT), 2, '0'), ''),
                                           'hex'
                                         ) AS embed_hash
                                  FROM   UNNEST(
                                           STRING_TO_ARRAY(
                                             ( REGEXP_MATCH(
                                                 -- Remove outer quotes if present
                                                 TRIM(BOTH '"' FROM child.embeds::TEXT),
                                                 $re$'hash'\s*:\s*\{'data'\s*:\s*\[([0-9,\s]+)\]$re$
                                               )
                                             )[1]                                         -- the capture group
                                             , ','
                                           )
                                         ) AS n
                                ) h
                          LEFT  JOIN farcaster.casts   c ON c.hash = h.embed_hash
                          LEFT  JOIN nindexer.profiles p ON p.fid  = c.fid
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
                -- 7) Final output for this thread
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
                          -- Try to replace URL first, if no replacement happens, append quote at end
                          CASE
                            WHEN st.text ~ 'https:\/\/warpcast\.com\/[^ ]+\/0x[0-9A-Fa-f]{8,}' THEN
                              -- URL exists, replace it
                              REGEXP_REPLACE(st.text, 'https:\/\/warpcast\.com\/[^ ]+\/0x[0-9A-Fa-f]{8,}', st.quote_text, 'g')
                            ELSE
                              -- No URL, append quote at end
                              CONCAT(st.text, ' ', st.quote_text)
                          END
                        ELSE st.text
                      END
                    ),
                    E'\n'
                    ORDER BY st.sort_path
                  ) AS thread_blob
                FROM sorted_thread st
                            )
                    END AS thread_blob,
                    
                    -- FIDs array
                    CASE 
                        WHEN ri.reply_count > 50000 THEN
                            -- For large threads, just return the root fid
                            ARRAY[ri.root_fid]
                        ELSE
                            -- For normal threads, get all FIDs ordered by popularity
                            (
                                WITH
                -------------------------------------------------------------------------------
                -- 1) First, get all casts in the thread
                -------------------------------------------------------------------------------
                all_thread_casts AS (
                  WITH RECURSIVE thread_builder AS (
                    -- Root
                    SELECT c.hash, c.parent_hash, c.text, c.embeds, c.fid, 
                           0 AS depth, ARRAY[c.hash] AS path, c.fid AS op_fid
                    FROM farcaster.casts c
                    WHERE c.hash = bt.hash
                    
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
                  -- Get top 5% most popular posts (minimum 1)
                  popular_extras AS (
                    SELECT hash
                    FROM (
                      SELECT
                        hash,
                        reaction_count,
                        ROW_NUMBER() OVER (ORDER BY reaction_count DESC) AS rn,
                        COUNT(*) OVER () AS total
                      FROM thread_with_data
                      WHERE reaction_count > 0  -- Only consider posts with reactions
                    ) s
                    WHERE rn <= GREATEST(1, CEIL(total * 0.05))  -- top 5%, but never 0 rows
                  ),
                  -- Get all ancestors of needed nodes using path arrays
                  all_needed AS (
                    SELECT DISTINCT ancestor_hash
                    FROM (
                      SELECT hash FROM op_interactions
                      UNION
                      SELECT hash FROM popular_extras  -- Include popular posts
                    ) x
                    JOIN thread_with_data t ON t.hash = x.hash
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
                      -- Parse quotes inline using regex to handle malformed JSON
                      CASE
                        WHEN wp.embeds::text <> '"[]"' 
                         AND wp.embeds::text LIKE '%castId%'                 -- fast pre-filter
                        THEN (
                          /* 1. Pull the first "…'hash': {'data': [ … ]}" numeric list with a regex
                             2. Split to individual numbers
                             3. int → hex → string_agg → DECODE → bytea  */
                          SELECT CONCAT('QUOTE:["', c.text, '" - @', p.username, ']')
                          FROM  LATERAL (
                                  SELECT DECODE(
                                           STRING_AGG(LPAD(TO_HEX(TRIM(n)::INT), 2, '0'), ''),
                                           'hex'
                                         ) AS embed_hash
                                  FROM   UNNEST(
                                           STRING_TO_ARRAY(
                                             ( REGEXP_MATCH(
                                                 -- Remove outer quotes if present
                                                 TRIM(BOTH '"' FROM wp.embeds::TEXT),
                                                 $re$'hash'\s*:\s*\{'data'\s*:\s*\[([0-9,\s]+)\]$re$
                                               )
                                             )[1]                                         -- the capture group
                                             , ','
                                           )
                                         ) AS n
                                ) h
                          LEFT  JOIN farcaster.casts   c ON c.hash = h.embed_hash
                          LEFT  JOIN nindexer.profiles p ON p.fid  = c.fid
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
                        WHEN child.embeds::text <> '"[]"' 
                         AND child.embeds::text LIKE '%castId%'                 -- fast pre-filter
                        THEN (
                          /* 1. Pull the first "…'hash': {'data': [ … ]}" numeric list with a regex
                             2. Split to individual numbers
                             3. int → hex → string_agg → DECODE → bytea  */
                          SELECT CONCAT('QUOTE:["', c.text, '" - @', p.username, ']')
                          FROM  LATERAL (
                                  SELECT DECODE(
                                           STRING_AGG(LPAD(TO_HEX(TRIM(n)::INT), 2, '0'), ''),
                                           'hex'
                                         ) AS embed_hash
                                  FROM   UNNEST(
                                           STRING_TO_ARRAY(
                                             ( REGEXP_MATCH(
                                                 -- Remove outer quotes if present
                                                 TRIM(BOTH '"' FROM child.embeds::TEXT),
                                                 $re$'hash'\s*:\s*\{'data'\s*:\s*\[([0-9,\s]+)\]$re$
                                               )
                                             )[1]                                         -- the capture group
                                             , ','
                                           )
                                         ) AS n
                                ) h
                          LEFT  JOIN farcaster.casts   c ON c.hash = h.embed_hash
                          LEFT  JOIN nindexer.profiles p ON p.fid  = c.fid
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
                -- 7) Get FIDs array ordered by popularity
                -------------------------------------------------------------------------------
                SELECT ARRAY_AGG(sub.fid ORDER BY sub.popularity DESC)
                FROM (
                  SELECT 
                    st2.fid,
                    MAX(st2.reaction_count) AS popularity
                  FROM sorted_thread st2
                  WHERE st2.fid IS NOT NULL
                  GROUP BY st2.fid
                ) sub
                            )
                    END AS fids_array,
                    
                    -- Total reactions
                    CASE 
                        WHEN ri.reply_count > 50000 THEN
                            -- For large threads, just get reactions for the root post
                            (
                                SELECT COUNT(*)
                                FROM farcaster.reactions r
                                JOIN farcaster.user_labels ul ON ul.target_fid = r.fid AND ul.label_value::int = 2
                                WHERE r.target_hash = bt.hash
                            )
                        ELSE
                            -- For normal threads, calculate total reactions
                            (
                                WITH
                -------------------------------------------------------------------------------
                -- Same CTEs as above, but just getting the total reactions
                -------------------------------------------------------------------------------
                all_thread_casts AS (
                  WITH RECURSIVE thread_builder AS (
                    -- Root
                    SELECT c.hash, c.parent_hash, c.text, c.embeds, c.fid, 
                           0 AS depth, ARRAY[c.hash] AS path, c.fid AS op_fid
                    FROM farcaster.casts c
                    WHERE c.hash = bt.hash
                    
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

                thread_reactions AS (
                  SELECT 
                    r.target_hash AS hash,
                    COUNT(*) AS reaction_count
                  FROM farcaster.reactions r
                  JOIN farcaster.user_labels ul ON ul.target_fid = r.fid AND ul.label_value::int = 2
                  WHERE r.target_hash IN (SELECT hash FROM all_thread_casts)
                  GROUP BY r.target_hash
                )

                SELECT SUM(reaction_count)
                FROM thread_reactions
                            )
                    END AS total_reactions
                FROM root_info ri
            ) AS thread_data
        )
        
        -- Update the threads table with processed data
        UPDATE unbias.threads t
        SET 
            blob = pt.thread_blob,
            fids = pt.fids_array,
            reactions = pt.total_reactions,
            claimed_at = NOW(),
            threads_status = 1
        FROM processed_threads pt
        WHERE t.hash = pt.hash;

        GET DIAGNOSTICS v_rows = ROW_COUNT;

        ------------------------------------------------------------------
        -- 2. Nothing left? Commit, log, and exit
        ------------------------------------------------------------------
        IF v_rows = 0 THEN
            COMMIT;
            RAISE NOTICE 'Finished: no more threads to process. Total batches: %',
                         v_batches;
            RETURN;
        END IF;

        ------------------------------------------------------------------
        -- 3. Commit this batch, measure elapsed, log it
        ------------------------------------------------------------------
        COMMIT;

        v_batches  := v_batches + 1;
        v_elapsed  := round(EXTRACT(epoch FROM clock_timestamp() - v_start), 2);

        RAISE NOTICE 'Batch % → % threads processed (% s) [first timestamp: %]',
                     v_batches, v_rows, v_elapsed, v_first_timestamp;

        ------------------------------------------------------------------
        -- 4. Respect test limit, if any
        ------------------------------------------------------------------
        IF p_max_batches IS NOT NULL AND v_batches >= p_max_batches THEN
            RAISE NOTICE 'User-defined limit of % batches reached, stopping.',
                         p_max_batches;
            RETURN;
        END IF;
    END LOOP;
END;
$$; 