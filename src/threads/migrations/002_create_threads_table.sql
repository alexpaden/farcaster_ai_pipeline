DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1
    FROM information_schema.tables 
    WHERE table_schema = 'unbias' 
      AND table_name = 'threads'
  ) THEN
    -- Define the table structure without initial population
    CREATE TABLE unbias.threads (
        hash BYTEA PRIMARY KEY,
        blob TEXT NULL,
        summary TEXT NULL,
        fids INT[] NULL,
        embed_blob VECTOR(1536) NULL,
        embed_summary VECTOR(1536) NULL,
        reactions INT NULL DEFAULT 0,
        author_fid INT NULL,
        timestamp TIMESTAMP WITH TIME ZONE NULL, -- Renamed from created_at
        threads_status SMALLINT DEFAULT 0,     -- 0=Unclaimed, 1=Claimed, 2=Processed, 3=Failed
        claimed_at TIMESTAMP WITH TIME ZONE NULL, -- Timestamp when claimed by a worker
        spam SMALL INT -- denormalized spam label column
    );

  END IF;
END $$;
