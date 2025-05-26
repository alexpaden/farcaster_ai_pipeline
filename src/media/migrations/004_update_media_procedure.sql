CREATE OR REPLACE PROCEDURE farcaster.process_media_casts (
    p_batch_size  INT DEFAULT 100000,
    p_max_batches INT DEFAULT NULL
)
LANGUAGE plpgsql AS $$
DECLARE
    v_batch          INT  := 0;
    v_done           BOOL := FALSE;
    v_id             BIGINT;

    -- Variables for detailed batch statistics
    v_urls_inserted  INT;
    v_casts_updated  INT;
    v_urls_single    INT;
    v_casts_single   INT;
    v_elapsed        NUMERIC;
    v_start          TIMESTAMPTZ;
BEGIN
    ----------------------------------------------------------------
    -- Session-scope settings
    ----------------------------------------------------------------
    PERFORM set_config('jit', 'off', false);
    PERFORM set_config('synchronous_commit', 'off',  false);
    PERFORM set_config('parallel_setup_cost', '10',  false);
    PERFORM set_config('parallel_tuple_cost', '0.01', false);
    PERFORM set_config('work_mem',           '256MB', false);
    PERFORM set_config('max_parallel_workers_per_gather', '32',  false);

    WHILE NOT v_done LOOP
        ----------------------------------------------------------------
        -- Respect optional batch limit
        ----------------------------------------------------------------
        IF p_max_batches IS NOT NULL AND v_batch >= p_max_batches THEN
            RAISE NOTICE 'limit % batches reached – stopping', p_max_batches;
            EXIT;
        END IF;

        v_batch := v_batch + 1;
        v_start := clock_timestamp(); -- Start timer for the batch
        
        -- Initialize counters for this batch
        v_urls_inserted := 0;
        v_casts_updated := 0;

        ----------------------------------------------------------------
        -- 2. Main batch processing with explicit error handling
        ----------------------------------------------------------------
        
        -- First, get the batch of casts to process
        FOR v_id IN
            SELECT id
            FROM   farcaster.casts
            WHERE  media_status = 0
              AND  embeds IS NOT NULL
              AND  embeds <> '"[]"'
              AND  "timestamp" < clock_timestamp() - INTERVAL '1 hour'
            ORDER  BY "timestamp"
            LIMIT  p_batch_size
            FOR UPDATE SKIP LOCKED
        LOOP
            BEGIN
                -- Attempt to clean and parse the JSON
                WITH processed_json AS (
                    SELECT (
                        replace(
                            replace(
                                replace(
                                    substring(embeds::text, 2, length(embeds::text) - 2),
                                    '''', '"'
                                ),
                                '\"', '"'
                            ),
                            '`', ''''
                        )
                    )::jsonb AS json_data
                    FROM farcaster.casts
                    WHERE id = v_id
                ),
                extracted AS (
                    SELECT 
                        lower(e->>'url') AS url,
                        (e ? 'castId') AS has_cast
                    FROM processed_json,
                    LATERAL jsonb_array_elements(
                        CASE WHEN jsonb_typeof(json_data)='array'
                             THEN json_data
                             ELSE '[]'::jsonb
                        END
                    ) AS e
                ),
                urls_inserted AS (
                    INSERT INTO unbias.media (url)
                    SELECT url 
                    FROM extracted
                    WHERE url IS NOT NULL AND url <> ''
                    ON CONFLICT (url) DO NOTHING
                    RETURNING 1
                ),
                status_values AS (
                    SELECT 
                        max(CASE WHEN url IS NOT NULL AND url <> '' THEN 1 ELSE 0 END) AS has_url,
                        max(CASE WHEN has_cast THEN 1 ELSE 0 END) AS has_cast
                    FROM extracted
                ),
                update_status AS (
                    UPDATE farcaster.casts
                    SET media_status = CASE
                                         WHEN sv.has_url = 1 THEN 2
                                         WHEN sv.has_cast = 1 THEN 1
                                         ELSE 1
                                       END
                    FROM status_values sv
                    WHERE id = v_id
                    RETURNING 1
                )
                SELECT 
                    (SELECT count(*) FROM urls_inserted),
                    (SELECT count(*) FROM update_status)
                INTO v_urls_single, v_casts_single;
                
                -- Add to our counters
                v_urls_inserted := v_urls_inserted + v_urls_single;
                v_casts_updated := v_casts_updated + v_casts_single;
                
            EXCEPTION WHEN others THEN
                -- Any JSON errors or other issues, mark as status 5
                UPDATE farcaster.casts
                SET media_status = 5  -- 5 = corrupt/unparseable JSON
                WHERE id = v_id;
                
                -- Still count this as processed
                v_casts_updated := v_casts_updated + 1;
            END;
        END LOOP;

        ----------------------------------------------------------------
        -- 3. bookkeeping & exit test
        ----------------------------------------------------------------
        v_elapsed := round(EXTRACT(epoch FROM clock_timestamp() - v_start), 2);

        RAISE NOTICE 'batch %, urls_inserted=%, casts_updated=%, elapsed_s=%',
            v_batch, v_urls_inserted, v_casts_updated, v_elapsed;

        COMMIT;

        -- Only exit if we've processed less than a full batch (no more data)
        -- AND we haven't been asked to run a specific number of batches
        v_done := (v_casts_updated < p_batch_size) AND 
                  (p_max_batches IS NULL OR v_batch >= p_max_batches);
    END LOOP;
END;
$$;
