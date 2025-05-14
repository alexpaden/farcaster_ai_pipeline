CREATE OR REPLACE PROCEDURE process_casts(
    p_batch_size   int  DEFAULT 100000,
    p_max_batches  int  DEFAULT NULL          -- NULL = run to completion
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_batch    int := 0;
    v_updated  int;
    v_start    timestamptz;
    v_elapsed  numeric;
BEGIN
    ------------------------------------------------------------------
    -- Session‑only settings (revert automatically when session ends)
    ------------------------------------------------------------------
    PERFORM set_config('jit',                'off',  false);  -- session scope
    PERFORM set_config('synchronous_commit', 'off',  false);

<<main_loop>>
    LOOP
        IF p_max_batches IS NOT NULL AND v_batch >= p_max_batches THEN
            RAISE NOTICE 'User‑defined limit of % batches reached, stopping.',
                         p_max_batches;
            EXIT;
        END IF;

        v_batch := v_batch + 1;
        v_start := clock_timestamp();

        ------------------------------------------------------------------
        -- ONE statement does everything and returns the update count
        ------------------------------------------------------------------
        WITH upd AS (
            UPDATE farcaster.casts c
               SET threads_status = CASE
                                        WHEN c.hash = c.root_parent_hash
                                        THEN 2      -- root
                                        ELSE 1      -- reply
                                    END
             WHERE ctid IN (
                   SELECT ctid
                   FROM   farcaster.casts
                   WHERE  threads_status = 0
                     AND  "timestamp" < clock_timestamp() - INTERVAL '6 hours'
                   ORDER  BY "timestamp"
                   LIMIT  p_batch_size
             )
             RETURNING c.hash, c.fid, c."timestamp",
                       (c.hash = c.root_parent_hash) AS is_root
        ),
        ins AS (
            INSERT INTO unbias.threads (hash, author_fid, "timestamp",
                                        threads_status, claimed_at)
            SELECT hash, fid, "timestamp", 0, NULL
            FROM   upd
            WHERE  is_root
            ON CONFLICT DO NOTHING
        )
        SELECT COUNT(*) INTO v_updated   -- <‑‑ rows we just updated
        FROM   upd;

        v_elapsed := round(EXTRACT(epoch FROM clock_timestamp() - v_start), 2);

        RAISE NOTICE 'Batch % → % rows  (%.2f s)',
                     v_batch, v_updated, v_elapsed;

        COMMIT;

        IF v_updated < p_batch_size THEN
            RAISE NOTICE 'No more rows left after % batches.', v_batch;
            EXIT;
        END IF;
    END LOOP;
END;
$$;
