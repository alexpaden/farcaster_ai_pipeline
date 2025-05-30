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

# Assuming your db connection module is in the common src/db directory
# Adjust the import path if your project structure is different
from src.db.connect import db

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(worker)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DB_COMMAND_TIMEOUT = 300
VOYAGE_MODEL = "voyage-3.5-lite"
EMBEDDING_DIM = 512  # Must match database schema (vector(512) and halfvec(512))
# Voyage API limits: 1,000 texts or 1M tokens per request
# You can increase up to 1,000 if tracking token counts
DEFAULT_BATCH_SIZE = 128

class ThreadEmbeddingWorker:
    """Worker class for processing thread embeddings"""
    
    def __init__(self, worker_id: int, voyage_client: voyageai.AsyncClient, 
                 batch_size: int = 128, max_batches: int = 1, test_mode: bool = False):
        self.worker_id = worker_id
        self.voyage_client = voyage_client
        self.batch_size = batch_size
        self.max_batches = max_batches
        self.test_mode = test_mode
        self.batches_processed = 0
        self.logger = logging.LoggerAdapter(logger, {'worker': f'Worker-{worker_id}'})
        
    async def claim_batch(self) -> List[Tuple[bytes, str]]:
        """Claim a batch of threads for processing
        
        Returns:
            List of (hash, blob) tuples where hash is bytes and blob is str
        """
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
    
    async def process_batch(self, batch: List[Tuple[bytes, str]]) -> Tuple[List[Tuple[bytes, np.ndarray]], List[bytes]]:
        """Process a batch of texts and generate embeddings
        
        Args:
            batch: List of (hash, blob) tuples where hash is bytes
            
        Returns:
            Tuple of (successful, failed) where successful has (hash, embedding) pairs
            
        Note:
            Handles Voyage API rate limits (HTTP 429) with exponential backoff.
            Voyage returns 429 when hitting per-minute rate or concurrency limits.
            Our retry logic waits and retries up to max_retries times.
        """
        if not batch:
            return [], []
        
        hashes = [item[0] for item in batch]
        texts = [item[1] for item in batch]
        
        # Replace "↳" with "-" in memory
        processed_texts = [text.replace("↳", "-") if text else "" for text in texts]
        
        # Retry logic for API calls
        max_retries = 3
        retry_delay = 1.0  # Start with 1 second
        
        for attempt in range(max_retries):
            try:
                # Generate embeddings using VoyageAI
                self.logger.info(f"Generating embeddings for {len(processed_texts)} texts (attempt {attempt + 1}/{max_retries})")
                result = await self.voyage_client.embed(
                    processed_texts, 
                    model=VOYAGE_MODEL,
                    input_type="document",
                    output_dimension=512
                )
                
                # Convert embeddings to numpy arrays
                embeddings = [np.array(emb, dtype=np.float32) for emb in result.embeddings]
                
                # Log the actual embedding dimensions
                if embeddings:
                    actual_dim = len(embeddings[0])
                    self.logger.info(f"Embedding dimensions: {actual_dim} (expected: {EMBEDDING_DIM})")
                    if actual_dim != EMBEDDING_DIM:
                        self.logger.warning(f"Dimension mismatch! voyage-3.5-lite returned {actual_dim}D embeddings, but database expects {EMBEDDING_DIM}D")
                
                # Pair hashes with embeddings - hashes already bytes
                successful = [(hashes[i], embeddings[i]) for i in range(len(embeddings))]
                failed = []
                
                self.logger.info(f"Successfully generated {len(successful)} embeddings")
                return successful, failed
                
            except Exception as e:
                error_msg = str(e).lower()
                # Voyage API returns HTTP 429 when hitting rate/concurrency limits
                is_rate_limit = any(term in error_msg for term in ['rate limit', 'too many requests', '429'])
                
                if attempt < max_retries - 1:
                    # Determine wait time based on error type
                    if is_rate_limit:
                        wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                        self.logger.warning(f"Rate limit hit, waiting {wait_time}s before retry...")
                    else:
                        wait_time = retry_delay
                        self.logger.warning(f"API error: {e}, waiting {wait_time}s before retry...")
                    
                    await asyncio.sleep(wait_time)
                else:
                    # Final attempt failed
                    self.logger.error(f"Failed to generate embeddings after {max_retries} attempts: {e}")
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
        
        async with db.pool.acquire() as conn:
            # Register pgvector for this connection
            await pgvector.asyncpg.register_vector(conn)
            
            async with conn.transaction():
                # Process updates one by one to ensure proper type handling
                updated_count = 0
                for hash_val, embedding in successful:
                    # Convert numpy array to Python list
                    vector_data = embedding.tolist()
                    
                    # Update vector column first
                    result = await conn.execute("""
                        UPDATE unbias.threads
                        SET    blob_embedding = $2,
                               threads_status = 3
                        WHERE  hash = $1;
                    """, hash_val, vector_data)
                    
                    # Then update halfvec column separately
                    if result.endswith('1'):
                        await conn.execute("""
                            UPDATE unbias.threads
                            SET    blob_embedding_fp16 = blob_embedding
                            WHERE  hash = $1;
                        """, hash_val)
                        updated_count += 1
                
                # Row count sanity check
                if updated_count != len(successful):
                    self.logger.error(f"Row update mismatch: expected {len(successful)}, updated {updated_count}")
                else:
                    self.logger.info(f"Updated {updated_count} embeddings in database")
    
    async def mark_failed(self, failed_hashes: List[bytes]):
        """Mark failed items with status 4
        
        Args:
            failed_hashes: List of hash values as bytes objects
        """
        if not failed_hashes:
            return
        
        async with db.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("""
                    UPDATE unbias.threads
                    SET    threads_status = 4
                    WHERE  hash = ANY($1);
                """, failed_hashes)
                
                self.logger.info(f"Marked {len(failed_hashes)} items as failed")
    
    async def run(self):
        """Main worker loop"""
        self.logger.info(f"Starting worker {self.worker_id}")
        current_batch_hashes = []
        
        try:
            while self.batches_processed < self.max_batches:
                # Claim a batch
                batch = await self.claim_batch()
                
                if not batch:
                    self.logger.info("No more rows to process")
                    break
                
                self.logger.info(f"Claimed batch of {len(batch)} items")
                
                # Track current batch for crash recovery
                current_batch_hashes = [item[0] for item in batch]
                
                # Process the batch
                successful, failed = await self.process_batch(batch)
                
                # Update database
                await self.update_embeddings(successful)
                await self.mark_failed(failed)
                
                # Clear current batch after successful processing
                current_batch_hashes = []
                
                self.batches_processed += 1
                self.logger.info(f"Completed batch {self.batches_processed}/{self.max_batches}")
                
                # In test mode, exit after first batch
                if self.test_mode:
                    break
                    
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
        
        self.logger.info(f"Worker {self.worker_id} finished. Processed {self.batches_processed} batches")


async def main(args):
    """Main entry point for the threads embedding pipeline"""
    logger.info("Starting threads embedding pipeline...")
    start_time = asyncio.get_event_loop().time()
    
    # Check for VoyageAI API key
    if not os.getenv("VOYAGE_API_KEY"):
        logger.error("VOYAGE_API_KEY environment variable not set")
        return
    
    try:
        # Initialize the database connection pool
        await db.initialize_pool(command_timeout=DB_COMMAND_TIMEOUT)
        logger.info("Database pool initialized.")
        
        # Register pgvector extension with all connections in the pool
        async def setup_pgvector(connection):
            await pgvector.asyncpg.register_vector(connection)
        
        # Apply setup to all existing and future connections
        async with db.pool.acquire() as conn:
            await setup_pgvector(conn)
        
        # Set the setup function for new connections
        # Note: asyncpg doesn't have a direct way to set this after pool creation
        # So we'll keep the per-connection registration in update_embeddings
        logger.info("pgvector types registered.")
        
        # Run migrations
        module_migrations_path = str(Path(__file__).parent)
        await db.run_migrations(module_path=module_migrations_path)
        logger.info(f"Migrations for module '{module_migrations_path}' processed.")
        
        # Initialize VoyageAI client
        voyage_client = voyageai.AsyncClient()
        logger.info("VoyageAI client initialized")
        
        # Create workers
        workers = []
        for i in range(args.workers):
            worker = ThreadEmbeddingWorker(
                worker_id=i,
                voyage_client=voyage_client,
                batch_size=args.batch_size,
                max_batches=args.max_batches,
                test_mode=args.test
            )
            workers.append(worker)
        
        # Run workers in parallel
        logger.info(f"Starting {args.workers} workers with batch_size={args.batch_size}, max_batches={args.max_batches}")
        if args.test:
            logger.info("Running in TEST MODE - will process maximum 10 rows")
        
        await asyncio.gather(*[worker.run() for worker in workers])
        
        total_duration = asyncio.get_event_loop().time() - start_time
        logger.info(f"Threads embedding pipeline completed in {total_duration:.2f} seconds")
        
    except Exception as e:
        logger.error(f"An error occurred in the main workflow: {e}", exc_info=True)
    finally:
        # Ensure the database pool is closed
        if db._pool:
            await db.close_pool()
            logger.info("Database pool closed.")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Thread Embedding Batch Processor')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
                       help=f'Number of texts per batch (default: {DEFAULT_BATCH_SIZE}, max: 1000 per Voyage API)')
    parser.add_argument('--max-batches', type=int, default=1,
                       help='Maximum number of batches per worker (default: 1)')
    parser.add_argument('--workers', type=int, default=1,
                       help='Number of parallel workers')
    parser.add_argument('--test', action='store_true',
                       help='Run in test mode (process max 10 rows)')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
