"""
Database operations for the embedding pipeline.
"""

import time
import asyncio
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from .logger import PipelineLogger
from .utils import log_and_reraise

async def get_unprocessed_estimate(pool, last_id: int = 0) -> int:
    """Get current count of unprocessed rows."""
    async with pool.acquire() as conn:
        await conn.execute("SET statement_timeout = '60s'")
        if last_id == 0:
            result = await conn.fetchval("SELECT get_unprocessed_count()")
        else:
            result = await conn.fetchval("""
                SELECT count(*)
                FROM public.casts
                WHERE embedding384 IS NULL 
                AND id > $1
                AND text IS NOT NULL 
                AND length(trim(text)) > 0
            """, last_id)
        return int(result or 0)

async def fetch_next_batch(pool, last_id: int = 0, limit: int = 100, logger: Optional[PipelineLogger] = None) -> Tuple[List[Dict[str, Any]], float]:
    """Fetch next batch with detailed timing."""
    fetch_start = time.time()
    try:
        async with pool.acquire() as conn:
            await conn.execute("SET statement_timeout = '60s'")
            
            # Select query
            rows = await conn.fetch("""
                SELECT id, text
                FROM public.casts
                WHERE embedding384 IS NULL
                    AND id > $1
                    AND text IS NOT NULL
                    AND length(trim(text)) > 0
                ORDER BY id
                LIMIT $2
            """, last_id, limit)
            
            if not rows:
                return [], time.time() - fetch_start
                
            # Update timestamps
            ids = [row['id'] for row in rows]
            await conn.execute("""
                UPDATE casts
                SET embedding384_updated_at = NOW()
                WHERE id = ANY($1::bigint[])
            """, ids)
            
            return [dict(row) for row in rows], time.time() - fetch_start
    except Exception as e:
        log_and_reraise(e, "fetch_next_batch")

async def update_embeddings(pool, batch_ids: List[int], embeddings: np.ndarray, chunk_size: int, logger: Optional[PipelineLogger] = None, worker_id: int = 0) -> Tuple[float, int]:
    """Update embeddings with optimized batch processing."""
    start_time = time.time()
    total_updated = 0
    
    try:
        async with pool.acquire() as conn:
            # Verify function exists
            exists = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT 1 FROM pg_proc 
                    WHERE proname = 'batch_update_embeddings' 
                    AND pronamespace = 'public'::regnamespace
                )
            """)
            if not exists:
                raise RuntimeError("batch_update_embeddings function not found in public schema")
            
            for i in range(0, len(batch_ids), chunk_size):
                chunk_ids = batch_ids[i:i + chunk_size]
                chunk_embeddings = embeddings[i:i + chunk_size]
                chunk_start = time.time()
                
                try:
                    async with conn.transaction():
                        # Format vectors as strings for pgvector
                        vector_strings = [
                            f"[{','.join(map(lambda x: f'{float(x):.6f}', emb.tolist()))}]"
                            for emb in chunk_embeddings
                        ]
                        
                        # Use the optimized batch update function
                        await conn.execute(
                            "SELECT public.batch_update_embeddings($1::bigint[], $2::text[])",
                            chunk_ids,
                            vector_strings
                        )
                        
                        rows_updated = len(chunk_ids)
                        total_updated += rows_updated
                        chunk_time = time.time() - chunk_start
                        
                        if logger:
                            logger.log_db_chunk(chunk_time, rows_updated, len(chunk_ids))
                            
                        await asyncio.sleep(0.01)  # Reduced sleep time
                        
                except Exception as chunk_error:
                    print(f"\n[Worker {worker_id} | {time.strftime('%H:%M:%S')}] Vector sample: {vector_strings[0] if vector_strings else 'No vectors'}")
                    log_and_reraise(chunk_error, f"update_embeddings chunk {i//chunk_size + 1} (worker {worker_id})")
                
    except Exception as e:
        log_and_reraise(e, f"update_embeddings (worker {worker_id})")
        
    return time.time() - start_time, total_updated

async def db_update_worker(pool, queue: asyncio.Queue, chunk_size: int, logger: PipelineLogger, worker_id: int = 0):
    """Worker to handle async DB updates."""
    total_processed = 0
    start_time = time.time()
    
    try:
        print(f"\n[Worker {worker_id} | {time.strftime('%H:%M:%S')}] Starting DB update worker")
        while True:
            try:
                item = await queue.get()
                if item is None:  # Poison pill
                    break
                    
                batch_ids, embeddings = item
                update_time, rows_updated = await update_embeddings(
                    pool, batch_ids, embeddings, chunk_size, logger, worker_id
                )
                total_processed += rows_updated
                logger.update_processed(rows_updated)
                queue.task_done()
                
                # Log progress every 50k rows
                if total_processed % 50000 < rows_updated:
                    elapsed = time.time() - start_time
                    print(f"\n[Worker {worker_id} | {time.strftime('%H:%M:%S')}] Total rows updated: {total_processed:,} ({total_processed/elapsed:.1f} rows/sec average)")
                    
                await asyncio.sleep(0)
                
            except Exception as batch_error:
                log_and_reraise(batch_error, f"db_update_worker batch (worker {worker_id})")
                
    except Exception as e:
        log_and_reraise(e, f"db_update_worker (worker {worker_id})")
    finally:
        elapsed = time.time() - start_time
        print(f"\n[Worker {worker_id} | {time.strftime('%H:%M:%S')}] Completed {total_processed:,} updates in {elapsed/60:.1f}m ({total_processed/elapsed:.1f} rows/sec average)") 