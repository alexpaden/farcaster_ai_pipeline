CREATE OR REPLACE PROCEDURE unbias.sync_spam_labels(
        p_batch_size   int DEFAULT 100000,
        p_max_batches  int DEFAULT NULL          -- NULL = run to completion
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_rows      int;           -- rows updated in this batch
    v_batches   int := 0;      -- how many batches we’ve finished
    v_start     timestamptz;   -- batch start time
    v_elapsed   numeric;       -- elapsed seconds (rounded)
BEGIN
    LOOP
        ------------------------------------------------------------------
        -- 1.  Pull & lock up to p_batch_size rows that need a change
        ------------------------------------------------------------------
        v_start := clock_timestamp();

        WITH batch AS (
            SELECT t.ctid,
                   ul.label_value::SMALLINT AS new_spam
            FROM   unbias.threads      t
            JOIN   farcaster.user_labels ul
                   ON ul.label_type = 'spam'
                  AND ul.target_fid  = t.author_fid
            WHERE  t.spam IS DISTINCT FROM ul.label_value::SMALLINT
            LIMIT  p_batch_size
            FOR UPDATE SKIP LOCKED
        )
        UPDATE unbias.threads t
           SET spam = b.new_spam
        FROM batch b
        WHERE t.ctid = b.ctid;

        GET DIAGNOSTICS v_rows = ROW_COUNT;

        ------------------------------------------------------------------
        -- 2.  Nothing left?  Commit, log, and exit
        ------------------------------------------------------------------
        IF v_rows = 0 THEN
            COMMIT;
            RAISE NOTICE 'Finished: no more rows needed updates. Total batches: %',
                         v_batches;
            RETURN;
        END IF;

        ------------------------------------------------------------------
        -- 3.  Commit this batch, measure elapsed, log it
        ------------------------------------------------------------------
        COMMIT;

        v_batches  := v_batches + 1;
        v_elapsed  := round(EXTRACT(epoch FROM clock_timestamp() - v_start), 2);

        RAISE NOTICE 'Batch % → % rows (% s)',
                     v_batches, v_rows, v_elapsed;

        ------------------------------------------------------------------
        -- 4.  Respect test limit, if any
        ------------------------------------------------------------------
        IF p_max_batches IS NOT NULL AND v_batches >= p_max_batches THEN
            RAISE NOTICE 'User‑defined limit of % batches reached, stopping.',
                         p_max_batches;
            RETURN;
        END IF;
    END LOOP;
END;
$$;
