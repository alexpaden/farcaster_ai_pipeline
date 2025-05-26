DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1
    FROM information_schema.tables 
    WHERE table_schema = 'unbias' 
      AND table_name = 'media'
  ) THEN
    -- Define the table structure without initial population
    CREATE TABLE unbias.media (
        url TEXT PRIMARY KEY,
        summary TEXT NULL,
        media_type TEXT NULL,
        media_status SMALLINT DEFAULT 0,     -- 0=Unclaimed, 1=Claimed, 2=Processed, 3=Failed
        claimed_at TIMESTAMP WITH TIME ZONE NULL -- Timestamp when claimed by a worker
    );

  END IF;
END $$;
