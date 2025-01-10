"""
Main entry point for Farcaster AI pipelines.
Orchestrates the execution of various data processing and model pipelines.
"""

import os
import asyncio
import argparse
from pathlib import Path
from dotenv import load_dotenv
from embedding.pipeline import EmbeddingPipeline

async def run_migrations():
    """Run all database migrations in order."""
    load_dotenv()
    
    # Initialize pipeline to run migrations
    pipeline = EmbeddingPipeline(test_mode=False)
    await pipeline.initialize()
    await pipeline.pool.close()

async def run_embedding_pipeline(test_mode: bool = False, start_id: int = None):
    """Run the embedding pipeline with optional test mode and start ID."""
    pipeline = EmbeddingPipeline(test_mode=test_mode)
    await pipeline.initialize()
    
    try:
        await pipeline.process_casts(start_id)
    finally:
        await pipeline.pool.close()

async def main():
    parser = argparse.ArgumentParser(description='Farcaster AI Pipeline')
    parser.add_argument('--migrate-only', action='store_true', help='Only run migrations')
    parser.add_argument('--test', action='store_true', help='Run in test mode')
    parser.add_argument('--start-id', type=int, help='Resume from specific cast ID')
    args = parser.parse_args()
    
    if args.migrate_only:
        await run_migrations()
        return
    
    await run_embedding_pipeline(test_mode=args.test, start_id=args.start_id)

if __name__ == "__main__":
    asyncio.run(main()) 