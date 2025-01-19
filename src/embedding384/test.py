"""
Batch process embedding generation for casts table.
Uses async prefetching and parallel processing with TorchScript optimization and float16 precision on MPS.
Stores embeddings as int8 vectors for storage efficiency.
"""

import os
import time
import warnings
import sys
import asyncio
import multiprocessing
from pathlib import Path
from dotenv import load_dotenv

from src.db.connect import db
from .logger import PipelineLogger
from .batch import BatchProcessor, EmbeddingProcessor
from .db import get_unprocessed_estimate, db_update_worker
from .utils import log_and_reraise

# Configuration settings
MODEL_BATCH_SIZE = 2048  # Model inference batch size
BATCH_SIZE_ROWS = 20000  # Number of rows to process per instance
PREFETCH_BATCHES = 3  # Number of batches to prefetch
CHUNK_SIZE = 2000  # Larger chunks for bulk updates
DB_WORKERS = 1  # Number of parallel DB workers
DB_QUEUE_SIZE = 50  # Increased queue size for better parallelism
MAX_PENDING_UPDATES = 4000  # Stop inference when pending updates exceeds this

# Configure warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", message=".*_is_quantized_training_enabled.*")
warnings.filterwarnings("ignore", message=".*loss_type=None.*")

async def monitor_queues(db_queue: asyncio.Queue, logger: PipelineLogger):
    """Monitor queue sizes and processing rates."""
    while True:
        queue_size = db_queue.qsize()
        if queue_size > DB_QUEUE_SIZE * 0.8:
            print(f"\nWARNING: DB queue is {queue_size}/{DB_QUEUE_SIZE} full")
            print(f"Inference rate: {logger.get_inference_rate():.1f} rows/sec")
            print(f"DB update rate: {logger.get_db_rate():.1f} rows/sec")
        await asyncio.sleep(5)

async def process_casts(pool, model_batch_size: int):
    """Main processing orchestration with improved logging."""
    logger = PipelineLogger()
    total_remaining = await get_unprocessed_estimate(pool)
    logger.set_total_remaining(total_remaining)
    
    try:
        # Initialize processor and queues
        processor = BatchProcessor(pool, BATCH_SIZE_ROWS, PREFETCH_BATCHES, logger=logger)
        db_queue = asyncio.Queue(maxsize=DB_QUEUE_SIZE)
        logger.set_db_queue(db_queue)  # Set queue reference for monitoring
        await processor.start()
        
        # Start multiple DB workers
        db_workers = []
        for i in range(DB_WORKERS):
            worker = asyncio.create_task(
                db_update_worker(pool, db_queue, CHUNK_SIZE, logger, worker_id=i)
            )
            db_workers.append(worker)
        
        # Start batch monitoring
        async def monitor_batch():
            while True:
                await logger.update_batch_stats(pool)
                logger.log_batch_summary()
                await asyncio.sleep(30)  # Check every 30 seconds
                
        monitor_task = asyncio.create_task(monitor_batch())
        
        embedding_processor = EmbeddingProcessor(model_batch_size, BATCH_SIZE_ROWS, quiet=True, logger=logger)
        
        while True:
            # Check queue size before fetching next batch
            queue_size = db_queue.qsize()
            if queue_size > MAX_PENDING_UPDATES:  # Much lower threshold
                print(f"\nPausing inference - DB queue has {queue_size} pending updates")
                print(f"Inference rate: {logger.get_inference_rate():.1f} rows/sec")
                print(f"DB update rate: {logger.get_db_rate():.1f} rows/sec")
                
                # Wait until queue is nearly empty before resuming
                while db_queue.qsize() > MAX_PENDING_UPDATES // 2:
                    await asyncio.sleep(1)
                    if db_queue.qsize() % 1000 == 0:  # Log progress periodically
                        print(f"Still waiting - {db_queue.qsize()} updates pending")
                print("\nResuming inference - DB queue below threshold")

            result = await processor.get_next_batch()
            
            if not result:
                remaining = await get_unprocessed_estimate(pool, processor.last_id)
                if remaining == 0:
                    break
                await asyncio.sleep(0.1)  # Wait a bit before retrying
                continue
                
            batch, fetch_time = result
            texts = [cast['text'] for cast in batch]
            ids = [cast['id'] for cast in batch]
            
            # Process embeddings in smaller chunks for better pipelining
            chunk_size = min(len(texts), model_batch_size * 4)  # Process 4 batches at a time
            for i in range(0, len(texts), chunk_size):
                # Check queue size before processing each chunk
                if db_queue.qsize() > MAX_PENDING_UPDATES:
                    print(f"\nPausing inference - DB queue has {db_queue.qsize()} pending updates")
                    print(f"Inference rate: {logger.get_inference_rate():.1f} rows/sec")
                    print(f"DB update rate: {logger.get_db_rate():.1f} rows/sec")
                    
                    while db_queue.qsize() > MAX_PENDING_UPDATES // 2:
                        await asyncio.sleep(1)
                        if db_queue.qsize() % 1000 == 0:
                            print(f"Still waiting - {db_queue.qsize()} updates pending")
                    print("\nResuming inference - DB queue below threshold")

                chunk_texts = texts[i:i + chunk_size]
                chunk_ids = ids[i:i + chunk_size]
                
                # Process chunk
                embeddings, batch_ids, metrics_list = embedding_processor.process_batch(chunk_texts, chunk_ids)
                
                # Single queue attempt with backoff
                while True:
                    try:
                        await asyncio.wait_for(db_queue.put((batch_ids, embeddings)), timeout=1.0)
                        logger.queue_updates(len(batch_ids))
                        break
                    except asyncio.TimeoutError:
                        await asyncio.sleep(0.1)
            
    except Exception as e:
        log_and_reraise(e, "process_casts")
    finally:
        if 'processor' in locals():
            processor.done = True
            if processor._prefetch_task:
                await processor._prefetch_task
                
        # Cancel monitoring
        if 'monitor_task' in locals():
            monitor_task.cancel()
                
        # Wait for remaining DB updates
        if 'db_queue' in locals():
            if not db_queue.empty():
                print(f"\nWaiting for remaining DB updates to complete ({db_queue.qsize()} items in queue)...")
            
            # Send poison pills to all workers
            for _ in range(DB_WORKERS):
                await db_queue.put(None)
            
            # Wait for all workers to complete
            if 'db_workers' in locals():
                await asyncio.gather(*db_workers)

async def main():
    """Main processing orchestration."""
    try:
        start_time = time.time()
        print("\nStarting casts processing...")
        
        await db.initialize_pool()
        await db.run_migrations(module_path=str(Path(__file__).parent))
        
        # Get initial count of unprocessed rows
        total_unprocessed = await get_unprocessed_estimate(db.pool)
        print(f"Estimated unprocessed rows: {total_unprocessed:,}")
        
        await process_casts(db.pool, MODEL_BATCH_SIZE)
        
        total_duration = time.time() - start_time
        print(f"\nTotal processing time: {total_duration:.1f}s")
        
    finally:
        await db.close_pool()

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    asyncio.run(main()) 