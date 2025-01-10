import logging
from typing import List, Dict, Any, Optional
from psycopg2.extras import execute_values
from .connect import db
import time

logger = logging.getLogger(__name__)

def fetch_unprocessed_casts(batch_size: int) -> List[Dict]:
    """Fetch unprocessed casts with deadlock retry."""
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            with db.get_cursor() as cur:
                # Simplified query with direct update and return
                cur.execute("""
                    WITH to_update AS (
                        SELECT id, text
                        FROM public.test_casts
                        WHERE embedding IS NULL
                        AND embedding_started_at IS NULL
                        ORDER BY id
                        LIMIT %s
                        FOR UPDATE SKIP LOCKED
                    )
                    UPDATE public.test_casts t
                    SET embedding_started_at = NOW()
                    FROM to_update
                    WHERE t.id = to_update.id
                    RETURNING t.id, t.text
                """, (batch_size,))
                
                results = cur.fetchall()
                return [{'id': row[0], 'text': row[1]} for row in results]
                
        except Exception as e:
            if "deadlock detected" in str(e) and retry_count < max_retries - 1:
                retry_count += 1
                time.sleep(0.1 * retry_count)  # Exponential backoff
                continue
            raise
    
    return []

def update_cast_embeddings(embeddings_data: List[Dict]) -> None:
    """Update cast embeddings using execute_values for better performance."""
    if not embeddings_data:
        return
        
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            with db.get_cursor() as cur:
                # Use execute_values instead of execute_batch
                execute_values(cur, """
                    UPDATE public.test_casts
                    SET 
                        embedding = v.embedding::vector,
                        embedding_updated_at = NOW(),
                        embedding_started_at = NULL
                    FROM (VALUES %s) AS v(id, embedding)
                    WHERE test_casts.id = v.id
                """, [(data['cast_id'], data['embedding']) for data in embeddings_data])
                return
                
        except Exception as e:
            if "deadlock detected" in str(e) and retry_count < max_retries - 1:
                retry_count += 1
                time.sleep(0.1 * retry_count)  # Exponential backoff
                continue
            raise

def get_embedding_progress() -> Dict:
    """Get embedding progress."""
    with db.get_cursor() as cur:
        cur.execute("""
            SELECT 
                COUNT(*) as total,
                COUNT(CASE WHEN embedding IS NOT NULL THEN 1 END) as processed,
                COUNT(CASE WHEN embedding IS NULL THEN 1 END) as remaining
            FROM public.test_casts
        """)
        row = cur.fetchone()
        return {
            'total': row[0],
            'processed': row[1],
            'remaining': row[2]
        }

def mark_failed_processing(cast_ids: List[int], error: str) -> None:
    """
    Mark casts as failed processing with error message and reset processing flag.
    
    Args:
        cast_ids: List of cast IDs that failed
        error: Error message describing the failure
    """
    try:
        with db.get_cursor() as cur:
            execute_values(cur, """
                UPDATE public.test_casts 
                SET 
                    embedding_started_at = NULL,
                    last_error = v.error,
                    error_count = COALESCE(error_count, 0) + 1
                FROM (VALUES %s) AS v(id, error)
                WHERE test_casts.id = v.id
            """, [(cast_id, error) for cast_id in cast_ids], template='(%s, %s)')
            
        logger.warning(f"Marked {len(cast_ids)} casts as failed processing: {error}")
    except Exception as e:
        logger.error(f"Error marking failed processing: {str(e)}")
        raise 