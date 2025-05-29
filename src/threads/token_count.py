import asyncio
import logging
import time
from typing import AsyncIterator, List
import tiktoken
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from functools import partial

# Import the database connection
from src.db.connect import db

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
BATCH_SIZE = 25000  # Process rows in batches
MAX_WORKERS = multiprocessing.cpu_count()  # Use all available CPU cores
DB_COMMAND_TIMEOUT = 600  # 10 minutes for large queries

def count_tokens_batch(texts: List[str], encoding_name: str = "cl100k_base") -> int:
    """Count tokens for a batch of texts using tiktoken."""
    encoding = tiktoken.get_encoding(encoding_name)
    total_tokens = 0
    
    for text in texts:
        if text:  # Skip None/empty texts
            # Disable special token checks to handle texts with tokens like <|endoftext|>
            tokens = encoding.encode(text, disallowed_special=())
            total_tokens += len(tokens)
    
    return total_tokens

async def stream_threads_data() -> AsyncIterator[List[str]]:
    """Stream thread data from database in batches."""
    query = """
        SELECT blob
        FROM unbias.threads
        WHERE threads_status = 1 AND spam = 2
        ORDER BY timestamp DESC
    """
    
    async with db.pool.acquire() as conn:
        # Use cursor for streaming large result sets
        async with conn.transaction():
            cursor = await conn.cursor(query)
            
            while True:
                # Fetch BATCH_SIZE rows at a time
                rows = await cursor.fetch(BATCH_SIZE)
                
                if not rows:
                    break
                
                # Extract blob values from rows
                batch = [row['blob'] for row in rows]
                yield batch

async def count_tokens_parallel():
    """Count tokens for all threads using parallel processing."""
    logger.info("Starting token counting process...")
    start_time = time.time()
    
    total_tokens = 0
    total_rows = 0
    batch_count = 0
    
    # Create process pool for parallel token counting
    loop = asyncio.get_event_loop()
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        logger.info(f"Using {MAX_WORKERS} workers for parallel processing")
        
        # Process batches as they stream from database
        async for batch in stream_threads_data():
            batch_count += 1
            batch_size = len(batch)
            total_rows += batch_size
            
            # Count tokens in parallel
            batch_start = time.time()
            tokens = await loop.run_in_executor(
                executor,
                count_tokens_batch,
                batch
            )
            batch_time = time.time() - batch_start
            
            total_tokens += tokens
            
            # Log progress every 10 batches
            if batch_count % 10 == 0:
                elapsed = time.time() - start_time
                rate = total_rows / elapsed
                logger.info(
                    f"Processed {total_rows:,} rows in {batch_count} batches | "
                    f"Total tokens: {total_tokens:,} | "
                    f"Rate: {rate:.0f} rows/sec | "
                    f"Last batch: {batch_time:.2f}s"
                )
    
    return total_tokens, total_rows

async def main():
    """Main entry point."""
    logger.info("Initializing database connection...")
    
    try:
        # Initialize database pool with higher timeout for large queries
        await db.initialize_pool(
            min_size=1,
            max_size=4,  # Limited pool size since we're CPU-bound, not IO-bound
            command_timeout=DB_COMMAND_TIMEOUT
        )
        
        # Get row count first for progress tracking
        logger.info("Getting total row count...")
        async with db.pool.acquire() as conn:
            row_count = await conn.fetchval("""
                SELECT COUNT(*)
                FROM unbias.threads
                WHERE threads_status = 1 AND spam = 2
            """)
        
        logger.info(f"Found {row_count:,} rows to process")
        
        if row_count == 0:
            logger.warning("No rows found matching criteria")
            return
        
        # Count tokens
        start_time = time.time()
        total_tokens, processed_rows = await count_tokens_parallel()
        duration = time.time() - start_time
        
        # Print results
        logger.info("=" * 60)
        logger.info("TOKEN COUNT RESULTS")
        logger.info("=" * 60)
        logger.info(f"Total rows processed: {processed_rows:,}")
        logger.info(f"Total tokens: {total_tokens:,}")
        logger.info(f"Average tokens per row: {total_tokens/processed_rows:.2f}")
        logger.info(f"Processing time: {duration:.2f} seconds")
        logger.info(f"Processing rate: {processed_rows/duration:.0f} rows/second")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Error during token counting: {e}", exc_info=True)
        raise
    finally:
        # Clean up
        if db._pool:
            await db.close_pool()
            logger.info("Database connection closed")

if __name__ == "__main__":
    # Set multiprocessing start method for macOS compatibility
    multiprocessing.set_start_method('spawn', force=True)
    
    # Run the main function
    asyncio.run(main())
