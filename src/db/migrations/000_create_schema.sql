-- Create the unbias schema for general pipeline operations
CREATE SCHEMA IF NOT EXISTS unbias;

-- Create the public schema if it doesn't exist
CREATE SCHEMA IF NOT EXISTS public;

-- Create the vector extension if it doesn't exist
CREATE EXTENSION IF NOT EXISTS vector;

CREATE EXTENSION IF NOT EXISTS pgai CASCADE;

CREATE EXTENSION IF NOT EXISTS pg_tiktoken;

-- Set search path to include unbias schema
--ALTER DATABASE neynar_parquet_importer SET search_path TO unbias, farcaster, nindexer, public;

-- UPDATE farcaster.casts
-- SET text = NULLIF(TRIM(text), '')
-- WHERE text IS NOT NULL;