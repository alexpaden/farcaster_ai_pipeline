ALTER SYSTEM SET max_worker_processes = 16;

-- 0 is default, 1 is comment, 2 is original cast (thread) unprocessed, 3 is original cast (thread) processed
ALTER TABLE farcaster.casts
  ADD COLUMN threads_status SMALLINT DEFAULT 0;


