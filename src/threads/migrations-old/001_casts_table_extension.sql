-- 0 is default, 1 is comment, 2 is original cast (thread) (moved to unbias.threads)
ALTER TABLE farcaster.casts
  ADD COLUMN IF NOT EXISTS threads_status SMALLINT DEFAULT 0;


