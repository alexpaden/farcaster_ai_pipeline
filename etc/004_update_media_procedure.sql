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

        ----------------------------------------------------------------
        -- 1. QA pass for malformed JSON - commented out as not needed
        -- All JSON appears to be well-formed in the expected patterns:
        -- "[{'url': 'url_here'}]" or "[{'url': 'url1'}, {'url': 'url2'}]"
        ----------------------------------------------------------------
        /*
        FOR v_id IN
            SELECT id
            FROM   farcaster.casts
            WHERE  media_status = 0
              AND  embeds IS NOT NULL
              AND  embeds <> '"[]"'          -- skip the empty‑array string
            ORDER  BY "timestamp"
            LIMIT  p_batch_size
            FOR UPDATE SKIP LOCKED
        LOOP
            BEGIN
                -- attempt the cast; raises on malformed JSON
                PERFORM replace(
                           trim(both '"' from embeds::text),
                           '''','"'
                       )::jsonb
                FROM   farcaster.casts
                WHERE  id = v_id;

            EXCEPTION WHEN others THEN
                UPDATE farcaster.casts
                SET    media_status = 3       -- 3 = corrupt JSON
                WHERE  id = v_id;

                CONTINUE;                     -- next v_id
            END;                              -- close try/catch block
        END LOOP;                             -- close FOR v_id loop
        */

        ----------------------------------------------------------------
        -- 2. Main batch processing - now with single scan and JSON processing
        ----------------------------------------------------------------
        WITH casts_to_process AS (
            SELECT c.id,
                   replace(
                       trim(both '"' from c.embeds::text),
                       '''','"'
                   )::jsonb                AS embeds_json
            FROM   farcaster.casts c
            WHERE  c.media_status = 0
              AND  c.embeds <> '"[]"'       -- skip empty arrays
              AND  c."timestamp" < clock_timestamp() - INTERVAL '1 hour'
            ORDER  BY c."timestamp"
            LIMIT  p_batch_size
            FOR UPDATE SKIP LOCKED
        ),
        extracted AS (
            SELECT  c.id                AS cast_id,
                    lower(e->>'url')    AS url,
                    (e ? 'castId')      AS has_cast
            FROM    casts_to_process c,
                    LATERAL jsonb_array_elements(c.embeds_json) AS e
        ),
        -- Insert URLs first
        ins AS (
            INSERT INTO unbias.media (url)
            SELECT url
            FROM   extracted
            WHERE  url IS NOT NULL 
              AND  url <> ''
              --AND  url LIKE 'https://%'  -- Only accept https URLs
            ON CONFLICT (url) DO NOTHING
            RETURNING 1
        ),
        -- Calculate status using the same extracted data
        status AS (
            SELECT  c.id,
                    max(CASE WHEN e.url IS NOT NULL AND e.url <> '' THEN 1 ELSE 0 END) AS has_url,
                    max(CASE WHEN e.has_cast THEN 1 ELSE 0 END) AS has_cast
            FROM    casts_to_process c
            LEFT    JOIN extracted e ON e.cast_id = c.id
            GROUP   BY c.id
        ),
        -- Update casts with new status
        upd AS (
            UPDATE farcaster.casts fc
               SET media_status = CASE
                                     WHEN s.has_url  = 1 THEN 2
                                     WHEN s.has_cast = 1 THEN 1
                                     ELSE 1
                                   END
            FROM   status s
            WHERE  fc.id = s.id
            RETURNING 1
        )
        SELECT 
            (SELECT count(*) FROM ins) AS urls_inserted,
            (SELECT count(*) FROM upd) AS casts_updated
        INTO v_urls_inserted, v_casts_updated;

        ----------------------------------------------------------------
        -- 3. bookkeeping & exit test
        ----------------------------------------------------------------
        v_elapsed := round(EXTRACT(epoch FROM clock_timestamp() - v_start), 2);

        RAISE NOTICE 'batch %, urls_inserted=%, casts_updated=%, elapsed_s=%',
            v_batch, v_urls_inserted, v_casts_updated, v_elapsed;

        COMMIT;

        v_done := (v_casts_updated < p_batch_size);   -- true = tail reached
    END LOOP;
END;
$$;
