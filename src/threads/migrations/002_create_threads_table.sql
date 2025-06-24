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
        spam SMALLINT, -- denormalized spam label column
        blob TEXT NULL,
        fids INT[] NULL,
        reactions INT NULL DEFAULT 0,
        author_fid INT NULL,
        timestamp TIMESTAMP WITH TIME ZONE NULL, -- Renamed from created_at
        thread_status SMALLINT DEFAULT 0,     --  0=Unclaimed, 1=Claimed, ... etc
        blob_timestamp TIMESTAMP WITH TIME ZONE NULL, -- Formerly claimed_at
        blob_embedding VECTOR(512) NULL,
        blob_embedding_fp16 HALFVEC(512) NULL,
        blob_embedding_binary bit(512) NULL,
        classifier_status SMALLINT DEFAULT 0,
        classifier_timestamp TIMESTAMP WITH TIME ZONE NULL,
        thread_type VARCHAR(100) NULL,
        value_exchange VARCHAR(100) NULL,
        interaction_pattern VARCHAR(100) NULL,
        keywords_array VARCHAR(100)[] NULL,
        tokens INT NULL       
    );

  END IF;
END $$;
