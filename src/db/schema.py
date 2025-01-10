import logging
from .connect import db

logger = logging.getLogger(__name__)

def update_schema():
    """
    Update the database schema with necessary columns and indexes for the embedding pipeline.
    Uses IF NOT EXISTS to make it safe to run multiple times.
    """
    try:
        with db.get_cursor() as cur:
            # Enable pgvector extension if not enabled
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Add embedding column and metadata columns
            cur.execute("""
                ALTER TABLE public.casts
                ADD COLUMN IF NOT EXISTS embedding vector(1536),
                ADD COLUMN IF NOT EXISTS embedding_started_at timestamp with time zone,
                ADD COLUMN IF NOT EXISTS embedding_updated_at timestamp with time zone,
                ADD COLUMN IF NOT EXISTS last_error text,
                ADD COLUMN IF NOT EXISTS error_count integer DEFAULT 0;
            """)
            
            # Create indexes for better performance
            cur.execute("""
                -- Index for finding unprocessed rows
                CREATE INDEX IF NOT EXISTS idx_casts_unprocessed 
                ON public.casts (id)
                WHERE embedding IS NULL 
                AND text IS NOT NULL 
                AND trim(text) != '';
                
                -- Index for the embedding column
                CREATE INDEX IF NOT EXISTS idx_casts_embedding 
                ON public.casts USING ivfflat (embedding vector_cosine_ops)
                WHERE embedding IS NOT NULL;
                
                -- Index for monitoring in-progress rows
                CREATE INDEX IF NOT EXISTS idx_casts_in_progress
                ON public.casts (embedding_started_at)
                WHERE embedding IS NULL;
            """)
            
            logger.info("Schema updated successfully")
            
    except Exception as e:
        logger.error(f"Error updating schema: {str(e)}")
        raise

if __name__ == "__main__":
    update_schema() 