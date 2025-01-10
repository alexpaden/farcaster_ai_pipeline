#!/usr/bin/env python3
"""
Backfill script for processing texts with automatic scaling.
"""

import asyncio
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any

from src.embedding.scaling import ScalingManager
from src.db.queries import fetch_unprocessed_casts, update_cast_embeddings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/backfill.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

async def process_batch(
    manager: ScalingManager,
    texts: List[str],
    cast_ids: List[int],
    time_limit_minutes: float = 5.0
) -> int:
    """Process a batch of texts with automatic scaling."""
    try:
        # Get embeddings with auto-scaling
        embeddings = await manager.process_texts(texts, time_limit_minutes)
        
        # Update database
        embeddings_data = [
            {'cast_id': cast_id, 'embedding': embedding.tolist()}
            for cast_id, embedding in zip(cast_ids, embeddings)
        ]
        update_cast_embeddings(embeddings_data)
        
        return len(texts)
        
    except Exception as e:
        logger.error(f"Error processing batch: {str(e)}")
        return 0

async def run_backfill(
    batch_size: int = 10000,
    time_limit_minutes: float = 5.0,
    max_workers: int = 9
) -> None:
    """Run backfill with automatic scaling."""
    try:
        # Initialize scaling manager
        manager = ScalingManager(max_workers=max_workers)
        total_processed = 0
        
        while True:
            # Fetch unprocessed casts
            casts = fetch_unprocessed_casts(batch_size=batch_size)
            if not casts:
                break
                
            texts = [cast['text'] for cast in casts]
            cast_ids = [cast['id'] for cast in casts]
            
            # Process batch
            processed = await process_batch(
                manager,
                texts,
                cast_ids,
                time_limit_minutes
            )
            total_processed += processed
            
            logger.info(
                f"Processed {processed} texts "
                f"(Total: {total_processed})"
            )
            
        logger.info(f"Backfill complete. Total processed: {total_processed}")
        
    except Exception as e:
        logger.error(f"Backfill failed: {str(e)}")
        raise
    finally:
        # Ensure we scale back down
        await manager.scale_workers(1)

async def main():
    parser = argparse.ArgumentParser(description="Run embedding backfill")
    parser.add_argument("--batch-size", type=int, default=10000,
                       help="Batch size for processing")
    parser.add_argument("--time-limit", type=float, default=5.0,
                       help="Time limit in minutes before scaling up")
    parser.add_argument("--max-workers", type=int, default=9,
                       help="Maximum number of workers")
    args = parser.parse_args()
    
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    await run_backfill(
        batch_size=args.batch_size,
        time_limit_minutes=args.time_limit,
        max_workers=args.max_workers
    )

if __name__ == "__main__":
    asyncio.run(main()) 