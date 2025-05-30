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
import aiohttp
import json

# Assuming your db connection module is in the common src/db directory
# Adjust the import path if your project structure is different
from src.db.connect import db

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
    
    def __init__(self, worker_id: int, api_key: str, 
                 progress_tracker: ProgressTracker,
                 batch_size: int = 128, max_batches: int = 0, test_mode: bool = False,
                 api_delay: float = 0.1):
        self.worker_id = worker_id
        self.api_key = api_key
        self.progress_tracker = progress_tracker
        self.batch_size = batch_size
        self.max_batches = max_batches
        self.test_mode = test_mode
        self.api_delay = api_delay  # Delay between API requests in seconds
        self.batches_processed = 0
        self.consecutive_errors = 0
        self.logger = logging.LoggerAdapter(logger, {'worker': f'Worker-{worker_id}'})
        self.session = None  # Will be created in run()
        
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
    
    async def process_batch(self, batch: List[Tuple[bytes, str]]) -> Tuple[List[Tuple[bytes, np.ndarray]], List[bytes]]:
        """Process a batch of texts and generate embeddings
        
        Args:
            batch: List of (hash, blob) tuples where hash is bytes
            
        Returns:
            Tuple of (successful, failed) where successful has (hash, embedding) pairs
            
        Note:
            Uses direct API calls with configurable delays between requests.
            Handles Voyage API rate limits (HTTP 429) with exponential backoff.
        """
        if not batch:
            return [], []
        
        hashes = [item[0] for item in batch]
        texts = [item[1] for item in batch]
        
        # Replace "↳" with "-" and filter out empty texts
        # Voyage API doesn't accept empty strings
        valid_indices = []
        processed_texts = []
        empty_hashes = []
        
        for i, text in enumerate(texts):
            if text and text.strip():  # Check if text exists and is not just whitespace
                processed_text = text.replace("↳", "-")
                processed_texts.append(processed_text)
                valid_indices.append(i)
            else:
                # Track empty texts as failed
                empty_hashes.append(hashes[i])
        
        # If all texts are empty, return all as failed
        if not processed_texts:
            self.logger.warning(f"All {len(batch)} texts in batch were empty")
            return [], hashes
        
        # Log if we had to skip some empty texts
        if empty_hashes:
            self.logger.warning(f"Skipping {len(empty_hashes)} empty texts in batch of {len(batch)}")
        
        # Add delay before API request (except for first request)
        if self.batches_processed > 0 and self.api_delay > 0:
            # Randomize delay between 50% and 150% of configured delay
            random_delay = self.api_delay * random.uniform(0.5, 1.5)
            await asyncio.sleep(random_delay)
        
        # Retry logic for API calls
        max_retries = 3
        retry_delay = 1.0  # Start with 1 second
        
        url = "https://api.voyageai.com/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "input": processed_texts,
            "model": VOYAGE_MODEL,
            "input_type": "document",
            "output_dimension": 512
        }
        
        for attempt in range(max_retries):
            try:
                async with self.session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Extract embeddings from response
                        embeddings = []
                        for item in data["data"]:
                            embedding = np.array(item["embedding"], dtype=np.float32)
                            embeddings.append(embedding)
                        
                        # Verify dimensions on first successful batch (silently)
                        if embeddings and self.batches_processed == 0:
                            actual_dim = len(embeddings[0])
                            if actual_dim != EMBEDDING_DIM:
                                self.logger.error(f"Dimension mismatch! voyage-3.5-lite returned {actual_dim}D embeddings, but database expects {EMBEDDING_DIM}D")
                        
                        # Pair hashes with embeddings - only for valid indices
                        successful = [(hashes[valid_indices[i]], embeddings[i]) for i in range(len(embeddings))]
                        
                        # Combine empty texts with any other failures
                        failed = empty_hashes
                        
                        # Reset consecutive errors on success
                        self.consecutive_errors = 0
                        return successful, failed
                    
                    elif response.status == 429:
                        # Rate limit error
                        error_text = await response.text()
                        if attempt < max_retries - 1:
                            wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                            self.logger.warning(f"Rate limit hit (429), waiting {wait_time}s before retry...")
                            await asyncio.sleep(wait_time)
                        else:
                            raise Exception(f"Rate limit error after {max_retries} attempts: {error_text}")
                    
                    else:
                        # Other HTTP errors
                        error_text = await response.text()
                        raise Exception(f"API error {response.status}: {error_text}")
                        
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay
                    self.logger.warning(f"API error: {e}, waiting {wait_time}s before retry...")
                    await asyncio.sleep(wait_time)
                else:
                    # Final attempt failed
                    self.logger.error(f"Failed to generate embeddings after {max_retries} attempts: {e}")
                    self.consecutive_errors += 1
                    if self.consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                        self.logger.error(f"Too many consecutive API errors ({MAX_CONSECUTIVE_ERRORS}), shutting down")
                        raise SystemExit("Too many consecutive API errors")
                    # All items in batch failed - hashes already bytes
                    return [], hashes
        
        # Should never reach here, but just in case
        return [], hashes
    
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
    
    async def mark_failed(self, failed_hashes: List[bytes]):
        """Mark failed items with status 4
        
        Args:
            failed_hashes: List of hash values as bytes objects
        """
        if not failed_hashes:
            return
        
        try:
            async with db.pool.acquire() as conn:
                async with conn.transaction():
                    result = await conn.execute("""
                        UPDATE unbias.threads
                        SET    threads_status = 4
                        WHERE  hash = ANY($1);
                    """, failed_hashes)
                    
                    # Only log for debugging if many failures
                    if len(failed_hashes) > 100:
                        self.logger.warning(f"Marked {len(failed_hashes)} items as failed")
        except Exception as e:
            self.logger.error(f"Database error marking failures: {e}")
            # Don't increment consecutive errors for this operation
    
    async def run(self):
        """Main worker loop"""
        current_batch_hashes = []
        
        # Timing accumulators
        total_claim_time = 0
        total_api_time = 0
        total_update_time = 0
        total_batches = 0
        
        # Create aiohttp session for this worker
        self.session = aiohttp.ClientSession()
        
        try:
            # max_batches=0 means unlimited processing
            while self.max_batches == 0 or self.batches_processed < self.max_batches:
                # Time the claim operation
                claim_start = time.time()
                batch = await self.claim_batch()
                claim_time = time.time() - claim_start
                total_claim_time += claim_time
                
                if not batch:
                    # No more rows to process
                    break
                
                # Track current batch for crash recovery
                current_batch_hashes = [item[0] for item in batch]
                
                # Time the API call
                api_start = time.time()
                successful, failed = await self.process_batch(batch)
                api_time = time.time() - api_start
                total_api_time += api_time
                
                # Time the database updates
                update_start = time.time()
                await self.update_embeddings(successful)
                await self.mark_failed(failed)
                update_time = time.time() - update_start
                total_update_time += update_time
                
                # Clear current batch after successful processing
                current_batch_hashes = []
                
                self.batches_processed += 1
                total_batches += 1
                
                # Log timing info for every batch
                avg_claim = total_claim_time / total_batches
                avg_api = total_api_time / total_batches
                avg_update = total_update_time / total_batches
                total_per_batch = avg_claim + avg_api + avg_update
                
                self.logger.info(
                    f"Worker {self.worker_id} timing (avg per batch): "
                    f"claim={avg_claim:.2f}s, api={avg_api:.2f}s, update={avg_update:.2f}s, "
                    f"total={total_per_batch:.2f}s ({self.batch_size/total_per_batch:.0f} rows/sec)"
                )
                
                # In test mode, exit after first batch
                if self.test_mode:
                    break
                    
        except SystemExit:
            # Re-raise to propagate shutdown
            raise
        except Exception as e:
            self.logger.error(f"Worker {self.worker_id} crashed: {e}", exc_info=True)
            raise
        finally:
            # If we have a batch in progress, mark it as failed
            if current_batch_hashes:
                self.logger.warning(f"Worker {self.worker_id} crashed with {len(current_batch_hashes)} items in progress, marking as failed")
                try:
                    await self.mark_failed(current_batch_hashes)
                except Exception as e:
                    self.logger.error(f"Failed to mark crashed batch as failed: {e}")
            
            # Clean up aiohttp session
            if self.session:
                await self.session.close()


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
        
        # Create progress tracker
        progress_tracker = ProgressTracker()
        
        # Get API key
        api_key = os.getenv("VOYAGE_API_KEY")
        if not api_key:
            logger.error("VOYAGE_API_KEY environment variable not set")
            return
        
        # Create workers
        workers = []
        for i in range(args.workers):
            worker = ThreadEmbeddingWorker(
                worker_id=i,
                api_key=api_key,
                progress_tracker=progress_tracker,
                batch_size=args.batch_size,
                max_batches=args.max_batches,
                test_mode=args.test,
                api_delay=args.api_delay
            )
            workers.append(worker)
        
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
    parser.add_argument('--api-delay', type=float, default=0.1,
                       help='Delay between API requests in seconds (default: 0.1)')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
