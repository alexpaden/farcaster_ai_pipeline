-- 1. B‑tree on author_fid
CREATE INDEX CONCURRENTLY idx_threads_author_fid
  ON unbias.threads (author_fid);

-- 2. GIN on fids array
CREATE INDEX CONCURRENTLY idx_threads_fids_gin
  ON unbias.threads
  USING GIN (fids);
