-- 0 is default, 1 is quoteId, 2 is media cast (+ inserted into unbias.media)
ALTER TABLE farcaster.casts
  ADD COLUMN IF NOT EXISTS media_status SMALLINT DEFAULT 0;

