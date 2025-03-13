-- Create the unbias schema for general pipeline operations
CREATE SCHEMA IF NOT EXISTS unbias;

-- Create the public schema if it doesn't exist
CREATE SCHEMA IF NOT EXISTS public;

-- Set search path to include unbias schema
ALTER DATABASE neynar_parquet_importer SET search_path TO unbias, farcaster, nindexer, public;