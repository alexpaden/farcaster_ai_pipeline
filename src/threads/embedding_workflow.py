import asyncio
import logging
from pathlib import Path
import os
import argparse
from typing import List, Tuple, Optional
import voyageai
import numpy as np
from contextlib import asynccontextmanager
import pgvector.asyncpg
import time
import random

# Assuming your db connection module is in the common src/db directory
# Adjust the import path if your project structure is different
from src.db.connect import db

# threads_status values: 1=ready/retry, 2=processing, 3=done, 4=api_failed, 5=blank_text

# Configure logging - default to INFO for progress updates
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Reduce voyage library verbosity while keeping our progress logs
logging.getLogger('voyage').setLevel(logging.WARNING)

# Constants
DB_COMMAND_TIMEOUT = 300
VOYAGE_MODEL = "voyage-3.5-lite"
EMBEDDING_DIM = 512  # Must match database schema (vector(512) and halfvec(512))
# Voyage API limits: 1,000 texts or 1M tokens per request
# You can increase up to 1,000 if tracking token counts
DEFAULT_BATCH_SIZE = 128
DEFAULT_MAX_BATCHES = 0  # 0 means unlimited - process all available rows
PROGRESS_LOG_INTERVAL = 10_000  # Log progress every 10k rows
MAX_CONSECUTIVE_ERRORS = 10  # Kill script after this many consecutive errors

class ProgressTracker:
    """Shared progress tracker for all workers"""
    def __init__(self):
        self.total_rows = 0
        self.last_milestone = 0
        self.start_time = time.time()
        self.lock = asyncio.Lock()
        self.first_update = True
        
    async def add_rows(self, count: int):
        """Add processed rows and log if milestone reached"""
        async with self.lock:
            self.total_rows += count
            
            # Log first update to show system is working
            if self.first_update:
                self.first_update = False
                elapsed = time.time() - self.start_time
                logger.info(f"Processing started: {self.total_rows:,} rows completed in {elapsed:.1f}s")
            
            # Check if we've crossed a milestone
            current_milestone = (self.total_rows // PROGRESS_LOG_INTERVAL) * PROGRESS_LOG_INTERVAL
            if current_milestone > self.last_milestone:
                elapsed = time.time() - self.start_time
                rate = self.total_rows / elapsed if elapsed > 0 else 0
                # Use INFO level for progress updates
                logger.info(f"Progress: {self.total_rows:,} rows processed in {elapsed:.1f}s ({rate:.0f} rows/sec)")
                self.last_milestone = current_milestone

class ThreadEmbeddingWorker:
    """Worker class for processing thread embeddings"""
    
    def __init__(self, worker_id: int, voyage_client: voyageai.AsyncClient, 
                 progress_tracker: ProgressTracker,
                 batch_size: int = 128, max_batches: int = 0, test_mode: bool = False):
        self.worker_id = worker_id
        self.voyage_client = voyage_client
        self.progress_tracker = progress_tracker
        self.batch_size = batch_size
        self.max_batches = max_batches
        self.test_mode = test_mode
        self.batches_processed = 0
        self.consecutive_errors = 0
        self.logger = logging.LoggerAdapter(logger, {'worker': f'Worker-{worker_id}'})
        
    async def claim_batch(self) -> List[Tuple[bytes, str]]:
        """Claim a batch of threads for processing
        
        Returns:
            List of (hash, blob) tuples where hash is bytes and blob is str
        """
        try:
            async with db.pool.acquire() as conn:
                async with conn.transaction():
                    # Use test limit if in test mode
                    limit = min(10, self.batch_size) if self.test_mode else self.batch_size
                    
                    # Claim batch with FOR UPDATE SKIP LOCKED for parallel processing
                    rows = await conn.fetch("""
                        WITH batch AS (
                            SELECT hash, blob
                            FROM   unbias.threads
                            WHERE  threads_status = 1
                              AND  spam = 2
                            ORDER  BY timestamp DESC
                            LIMIT  $1
                            FOR UPDATE SKIP LOCKED
                        )
                        UPDATE unbias.threads t
                        SET    threads_status = 2
                        FROM   batch b
                        WHERE  t.hash = b.hash
                        RETURNING b.hash, b.blob;
                    """, limit)
                    
                    # Return with consistent types - hash is already memoryview/bytes from asyncpg
                    # Convert memoryview to bytes for consistency
                    return [(bytes(row['hash']), row['blob']) for row in rows]
        except Exception as e:
            self.logger.error(f"Database error claiming batch: {e}")
            self.consecutive_errors += 1
            if self.consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                self.logger.error(f"Too many consecutive errors ({MAX_CONSECUTIVE_ERRORS}), shutting down")
                raise SystemExit("Too many consecutive database errors")
            raise
    
    async def process_batch(self, batch: List[Tuple[bytes, str]]) -> Tuple[List[Tuple[bytes, np.ndarray]], List[bytes], List[bytes], List[bytes]]:
        """Process a batch of texts and generate embeddings
        
        Returns:
            Tuple of (successful, failed, blank, retry) where successful has (hash, embedding) pairs,
            failed are hashes that failed API permanently, blank are hashes with empty text, retry are hashes to be reset to status 1
        """
        if not batch:
            return [], [], [], []
        
        hashes = [item[0] for item in batch]
        texts = [item[1] for item in batch]
        
        valid_indices = []
        processed_texts = []
        blank_hashes = []
        
        for i, text in enumerate(texts):
            if text and text.strip():
                processed_text = text.replace("↳", "-")
                processed_texts.append(processed_text)
                valid_indices.append(i)
            else:
                blank_hashes.append(hashes[i])
        
        if not processed_texts:
            self.logger.warning(f"All {len(batch)} texts in batch were empty")
            return [], [], hashes, []
        
        if blank_hashes:
            self.logger.warning(f"Skipping {len(blank_hashes)} empty texts in batch of {len(batch)}")
        
        max_retries = 3
        retry_delay = 1.0
        for attempt in range(max_retries):
            try:
                result = await self.voyage_client.embed(
                    processed_texts, 
                    model=VOYAGE_MODEL,
                    input_type="document",
                    output_dimension=512
                )
                embeddings = [np.array(emb, dtype=np.float32) for emb in result.embeddings]
                if embeddings and self.batches_processed == 0:
                    actual_dim = len(embeddings[0])
                    if actual_dim != EMBEDDING_DIM:
                        self.logger.error(f"Dimension mismatch! voyage-3.5-lite returned {actual_dim}D embeddings, but database expects {EMBEDDING_DIM}D")
                successful = [(hashes[valid_indices[i]], embeddings[i]) for i in range(len(embeddings))]
                failed = []
                self.consecutive_errors = 0
                return successful, failed, blank_hashes, []
            except Exception as e:
                error_msg = str(e).lower()
                is_rate_limit = any(term in error_msg for term in ['rate limit', 'too many requests', '429'])
                is_server_overload = any(term in error_msg for term in ['server is overloaded', 'not ready yet', 'server error', '503'])
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt) if is_rate_limit else retry_delay
                    await asyncio.sleep(wait_time)
                else:
                    self.logger.error(f"Failed to generate embeddings after {max_retries} attempts: {e}")
                    self.consecutive_errors += 1
                    if self.consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                        self.logger.error(f"Too many consecutive API errors ({MAX_CONSECUTIVE_ERRORS}), shutting down")
                        raise SystemExit("Too many consecutive API errors")
                    if is_server_overload:
                        # Mark for retry (status 1)
                        return [], [], blank_hashes, hashes
                    else:
                        # Permanent error
                        return [], hashes, blank_hashes, []
        return [], hashes, blank_hashes, []
    
    async def update_embeddings(self, successful: List[Tuple[bytes, np.ndarray]]):
        """Update database with successful embeddings
        
        Args:
            successful: List of (hash, embedding) tuples where hash is bytes
        """
        if not successful:
            return
        
        try:
            async with db.pool.acquire() as conn:
                # Register pgvector for this connection
                await pgvector.asyncpg.register_vector(conn)
                
                async with conn.transaction():
                    # Prepare data for bulk update
                    # Pass the embedding twice to avoid parameter reuse issues
                    update_data = []
                    for hash_val, embedding in successful:
                        # Convert numpy array to list for pgvector
                        vector_data = embedding.tolist()
                        # Pass vector_data twice - once for blob_embedding, once for blob_embedding_fp16
                        update_data.append((vector_data, vector_data, hash_val))
                    
                    # Bulk update using executemany
                    # This sends all updates in a single round trip to the database
                    await conn.executemany("""
                        UPDATE unbias.threads
                        SET blob_embedding = $1,
                            blob_embedding_fp16 = $2,
                            threads_status = 3
                        WHERE hash = $3;
                    """, update_data)
                    
                    # Update progress tracker
                    await self.progress_tracker.add_rows(len(successful))
                    
                    # Reset consecutive errors on success
                    self.consecutive_errors = 0
                    
        except Exception as e:
            self.logger.error(f"Database error updating embeddings: {e}")
            self.consecutive_errors += 1
            if self.consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                self.logger.error(f"Too many consecutive database errors ({MAX_CONSECUTIVE_ERRORS}), shutting down")
                raise SystemExit("Too many consecutive database errors")
            raise
    
    async def mark_failed(self, failed_hashes: List[bytes], status: int = 4):
        """Mark failed items with status 4 (API error), 5 (blank text), or 1 (retry)"""
        if not failed_hashes:
            return
        try:
            async with db.pool.acquire() as conn:
                async with conn.transaction():
                    await conn.execute(f"""
                        UPDATE unbias.threads
                        SET    threads_status = {status}
                        WHERE  hash = ANY($1);
                    """, failed_hashes)
                    if len(failed_hashes) > 100:
                        self.logger.warning(f"Marked {len(failed_hashes)} items as failed (status {status})")
        except Exception as e:
            self.logger.error(f"Database error marking failures: {e}")
    
    async def run(self):
        """Main worker loop"""
        current_batch_hashes = []
        total_claim_time = 0
        total_api_time = 0
        total_update_time = 0
        total_batches = 0
        try:
            while self.max_batches == 0 or self.batches_processed < self.max_batches:
                claim_start = time.time()
                batch = await self.claim_batch()
                claim_time = time.time() - claim_start
                total_claim_time += claim_time
                if not batch:
                    break
                current_batch_hashes = [item[0] for item in batch]
                api_start = time.time()
                successful, failed, blank, retry = await self.process_batch(batch)
                api_time = time.time() - api_start
                total_api_time += api_time
                update_start = time.time()
                await self.update_embeddings(successful)
                await self.mark_failed(failed, status=4)
                await self.mark_failed(blank, status=5)
                await self.mark_failed(retry, status=1)
                update_time = time.time() - update_start
                total_update_time += update_time
                current_batch_hashes = []
                self.batches_processed += 1
                total_batches += 1
                avg_claim = total_claim_time / total_batches
                avg_api = total_api_time / total_batches
                avg_update = total_update_time / total_batches
                total_per_batch = avg_claim + avg_api + avg_update
                self.logger.info(
                    f"Worker {self.worker_id} timing (avg per batch): "
                    f"claim={avg_claim:.2f}s, api={avg_api:.2f}s, update={avg_update:.2f}s, "
                    f"total={total_per_batch:.2f}s ({self.batch_size/total_per_batch:.0f} rows/sec)"
                )
                if self.test_mode:
                    break
        except SystemExit:
            raise
        except Exception as e:
            self.logger.error(f"Worker {self.worker_id} crashed: {e}", exc_info=True)
            raise
        finally:
            if current_batch_hashes:
                self.logger.warning(f"Worker {self.worker_id} crashed with {len(current_batch_hashes)} items in progress, marking as failed")
                try:
                    await self.mark_failed(current_batch_hashes, status=1)
                except Exception as e:
                    self.logger.error(f"Failed to mark crashed batch as failed: {e}")


async def main(args):
    """Main entry point for the threads embedding pipeline"""
    # Initial startup message at INFO level
    logger.info("Starting threads embedding pipeline...")
    start_time = time.time()
    
    # Check for VoyageAI API key
    if not os.getenv("VOYAGE_API_KEY"):
        logger.error("VOYAGE_API_KEY environment variable not set")
        return
    
    try:
        # Initialize the database connection pool
        await db.initialize_pool(command_timeout=DB_COMMAND_TIMEOUT)
        
        # Register pgvector extension with all connections in the pool
        async def setup_pgvector(connection):
            await pgvector.asyncpg.register_vector(connection)
        
        # Apply setup to all existing and future connections
        async with db.pool.acquire() as conn:
            await setup_pgvector(conn)
        
        # Run migrations
        module_migrations_path = str(Path(__file__).parent)
        await db.run_migrations(module_path=module_migrations_path)
        
        # Get API keys (comma-delimited)
        api_keys_str = os.getenv("VOYAGE_API_KEY")
        if not api_keys_str:
            logger.error("VOYAGE_API_KEY environment variable not set")
            return
        
        # Debug logging
        logger.info(f"Raw API keys string: {repr(api_keys_str)}")
        logger.info(f"Raw API keys string length: {len(api_keys_str)}")
        logger.info(f"First 50 chars: {api_keys_str[:50]}...")
        logger.info(f"Contains commas: {api_keys_str.count(',')}")
        logger.info(f"Contains semicolons: {api_keys_str.count(';')}")
        logger.info(f"Contains spaces: {api_keys_str.count(' ')}")
        
        # Parse API keys - try different delimiters
        if ',' in api_keys_str:
            api_keys = [key.strip() for key in api_keys_str.split(',') if key.strip()]
        elif ';' in api_keys_str:
            api_keys = [key.strip() for key in api_keys_str.split(';') if key.strip()]
        elif ' ' in api_keys_str and 'Bearer' not in api_keys_str:
            # Space-delimited, but not if it contains "Bearer" (single key with prefix)
            api_keys = [key.strip() for key in api_keys_str.split(' ') if key.strip()]
        else:
            # Single key
            api_keys = [api_keys_str.strip()]
            
        logger.info(f"Found {len(api_keys)} API key(s)")
        
        # Create progress tracker
        progress_tracker = ProgressTracker()
        
        # Create workers with different API keys
        workers = []
        for i in range(args.workers):
            # Rotate through available API keys
            api_key = api_keys[i % len(api_keys)]
            
            # Create a separate VoyageAI client for each worker
            voyage_client = voyageai.AsyncClient(api_key=api_key)
            
            worker = ThreadEmbeddingWorker(
                worker_id=i,
                voyage_client=voyage_client,
                progress_tracker=progress_tracker,
                batch_size=args.batch_size,
                max_batches=args.max_batches,
                test_mode=args.test
            )
            workers.append(worker)
            
            # Log which API key this worker is using (just the index for security)
            logger.info(f"Worker {i} using API key #{i % len(api_keys) + 1}")
        
        # Log configuration
        logger.info(f"Starting {args.workers} workers with batch_size={args.batch_size}, max_batches={args.max_batches}")
        if args.test:
            logger.info("Running in TEST MODE - will process maximum 10 rows")
        
        # Run workers in parallel
        try:
            await asyncio.gather(*[worker.run() for worker in workers])
        except SystemExit as e:
            logger.error(f"Pipeline shut down: {e}")
            raise
        
        # Final summary
        total_duration = time.time() - start_time
        total_rows = progress_tracker.total_rows
        rate = total_rows / total_duration if total_duration > 0 else 0
        logger.info(f"Threads embedding pipeline completed: {total_rows:,} rows in {total_duration:.1f}s ({rate:.0f} rows/sec)")
        
    except Exception as e:
        logger.error(f"An error occurred in the main workflow: {e}", exc_info=True)
    finally:
        # Ensure the database pool is closed
        if db._pool:
            await db.close_pool()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Thread Embedding Batch Processor')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
                       help=f'Number of texts per batch (default: {DEFAULT_BATCH_SIZE}, max: 1000 per Voyage API)')
    parser.add_argument('--max-batches', type=int, default=DEFAULT_MAX_BATCHES,
                       help='Maximum number of batches per worker (default: 0, unlimited)')
    parser.add_argument('--workers', type=int, default=1,
                       help='Number of parallel workers')
    parser.add_argument('--test', action='store_true',
                       help='Run in test mode (process max 10 rows)')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
