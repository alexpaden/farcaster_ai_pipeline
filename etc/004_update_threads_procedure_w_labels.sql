CREATE OR REPLACE PROCEDURE farcaster.process_casts(
    p_batch_size   int  DEFAULT 100000,
    p_max_batches  int  DEFAULT NULL          -- NULL = run to completion
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_batch    int := 0;
    v_elapsed  numeric;
BEGIN
    ------------------------------------------------------------------
    -- Session‑only settings (revert automatically when session ends)
    ------------------------------------------------------------------
    PERFORM set_config('jit',                'off',  false);  -- session scope
    PERFORM set_config('synchronous_commit', 'off',  false);
    PERFORM set_config('parallel_setup_cost', '10',  false);
    PERFORM set_config('parallel_tuple_cost', '0.01', false);
    PERFORM set_config('work_mem',           '256MB', false);
	PERFORM set_config('max_parallel_workers_per_gather', '32',  false);

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
        -- Optimized statement with EXISTS subquery instead of JOIN
        ------------------------------------------------------------------
        WITH casts_to_process AS (
            SELECT c.ctid
            FROM   farcaster.casts c
            WHERE  c.threads_status = 0
              AND  c."timestamp" < clock_timestamp() - INTERVAL '6 hours'
              AND  EXISTS (
                  SELECT 1
                  FROM   farcaster.user_labels ul
                  WHERE  ul.target_fid = c.fid
                    AND  ul.label_value = '2'
              )
            ORDER  BY c."timestamp"
            LIMIT  p_batch_size
            FOR    UPDATE SKIP LOCKED
        ),
        upd AS (
            UPDATE farcaster.casts c
               SET threads_status = CASE
                                        WHEN c.hash = c.root_parent_hash
                                        THEN 2      -- root
                                        ELSE 1      -- reply
                                    END
             WHERE ctid IN (SELECT ctid FROM casts_to_process)
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
        SELECT COUNT(*) INTO v_updated
        FROM   upd;

        v_elapsed := round(EXTRACT(epoch FROM clock_timestamp() - v_start), 2);

        RAISE NOTICE 'Batch % → % rows  (%.2f s)',
                     v_batch, v_updated, v_elapsed;

        COMMIT;

        IF v_updated < p_batch_size THEN
            RAISE NOTICE 'No more rows left after % batches.', v_batch;
            EXIT;
        END IF;
    END LOOP;
END;
$$;
