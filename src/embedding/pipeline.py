"""
Embedding pipeline for processing Farcaster casts using MiniLM-L6-v2.
Implements dynamic scaling and efficient batch processing.
"""

import os
import asyncio
import logging
import gc
import psutil
from typing import Optional, List, Tuple
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
import asyncpg
from pathlib import Path
import multiprocessing
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingPipeline:
    def __init__(self, test_mode: bool = False):
        self.test_mode = test_mode
        self.model = None
        self.pool = None
        self.batch_size = 256  # Fixed model batch size
        self.max_instances = 48  # Maximum parallel instances
        self.current_instances = 1  # Start with 1 instance
        self.device = torch.device("mps")
        self.process = psutil.Process()
        self.executor = ThreadPoolExecutor(max_workers=4)  # For parallel encoding
        
    async def initialize(self):
        """Initialize database connection and model."""
        # Load environment variables
        db_host = os.getenv("DB_HOST", "localhost")
        db_port = int(os.getenv("DB_PORT", "15432"))
        db_name = os.getenv("DB_NAME", "postgres")
        db_user = os.getenv("DB_USER", "postgres")
        db_pass = os.getenv("DB_PASSWORD", "")
        
        # Initialize database pool with larger size for parallel operations
        self.pool = await asyncpg.create_pool(
            host=db_host,
            port=db_port,
            database=db_name,
            user=db_user,
            password=db_pass,
            min_size=8,
            max_size=50,  # Support max instances + buffer
            command_timeout=60
        )
        
        # Run migrations first
        await self._run_migrations()
        
        # Initialize model
        logger.info("Initializing model on MPS...")
        self.model = SentenceTransformer(
            'sentence-transformers/all-MiniLM-L6-v2',
            device=str(self.device)
        )
        
        # Warm up with single batch
        logger.info("Warming up model...")
        warmup_texts = ["Warm-up sentence"] * self.batch_size
        _ = self.model.encode(
            warmup_texts,
            batch_size=self.batch_size,
            convert_to_tensor=True,
            device=self.device,
            normalize_embeddings=True
        )
        logger.info("Model initialization complete")
        
        # Log initial memory usage
        memory_mb = self.process.memory_info().rss / 1024 / 1024
        logger.info(f"Initial memory usage: {memory_mb:.1f}MB")

    async def _run_migrations(self):
        """Run all numbered migrations in order."""
        migrations_dir = Path("src/db/migrations")
        if not migrations_dir.exists():
            logger.warning(f"Migrations directory not found: {migrations_dir}")
            return

        # Get all numbered migration files
        migration_files = []
        for file in migrations_dir.glob("[0-9]*.sql"):
            # Extract migration number from filename
            if match := re.match(r"(\d+)_.*\.sql", file.name):
                number = int(match.group(1))
                migration_files.append((number, file))

        # Sort by migration number
        migration_files.sort(key=lambda x: x[0])
        
        # Run each migration in order
        async with self.pool.acquire() as conn:
            for number, file in migration_files:
                logger.info(f"Running migration {file.name}")
                try:
                    with open(file) as f:
                        sql = f.read()
                        await conn.execute(sql)
                    logger.info(f"Completed migration {file.name}")
                except Exception as e:
                    logger.error(f"Error in migration {file.name}: {str(e)}")
                    raise

    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024

    async def cleanup(self):
        """Clean up resources."""
        if self.pool:
            await self.pool.close()
        if self.model:
            del self.model
            torch.cuda.empty_cache()  # Clear CUDA cache if used
            gc.collect()  # Force garbage collection
        self.executor.shutdown()
        
    async def __aenter__(self):
        """Context manager entry."""
        await self.initialize()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.cleanup()

    def __del__(self):
        """Destructor to ensure cleanup."""
        if self.pool and not self.pool._closed:
            asyncio.create_task(self.cleanup())
    
    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode a batch of texts using the model."""
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_tensor=True,
            device=self.device,
            normalize_embeddings=True
        )
        # Convert to int8 for storage
        embeddings_np = embeddings.cpu().numpy()
        return (embeddings_np * 127).astype(np.int8)
    
    async def _process_batch(self, texts: List[str], ids: List[int], cast_ids: Optional[List[int]] = None) -> None:
        """Process a batch of texts and update database."""
        now = datetime.utcnow()
        
        try:
            # Run encoding in thread pool to not block event loop
            loop = asyncio.get_event_loop()
            embeddings_int8 = await loop.run_in_executor(
                self.executor, 
                self._encode_batch,
                texts
            )
            
            # Convert to pgvector format
            embeddings_list = [json.dumps(e.tolist()) for e in embeddings_int8]
            
            # Update database
            async with self.pool.acquire() as conn:
                if self.test_mode:
                    await conn.executemany(
                        """
                        UPDATE public.test_casts
                        SET embedding = $2::vector,
                            embedding_updated_at = $3,
                            error_count = 0,
                            last_error = NULL
                        WHERE id = $1
                        """,
                        [(id, emb, now) for id, emb in zip(ids, embeddings_list)]
                    )
                else:
                    await conn.executemany(
                        """
                        UPDATE public.casts
                        SET embedding = $2::vector,
                            embedding_updated_at = $3
                        WHERE id = $1
                        """,
                        [(id, emb, now) for id, emb in zip(ids, embeddings_list)]
                    )
        except Exception as e:
            # Log error and update error count
            error_msg = str(e)
            logger.error(f"Error processing batch: {error_msg}")
            
            if self.test_mode:
                async with self.pool.acquire() as conn:
                    await conn.executemany(
                        """
                        UPDATE public.test_casts
                        SET error_count = error_count + 1,
                            last_error = $2,
                            embedding_updated_at = $3
                        WHERE id = $1
                        """,
                        [(id, error_msg[:500], now) for id in ids]
                    )
            raise
    
    async def process_casts(self, start_id: Optional[int] = None):
        """Process all casts with dynamic scaling."""
        while True:
            # Calculate total texts to fetch based on current instances
            fetch_size = self.batch_size * self.current_instances * 2  # Fetch extra to reduce DB calls
            
            # Fetch unprocessed casts
            async with self.pool.acquire() as conn:
                if self.test_mode:
                    where_clause = "WHERE embedding IS NULL AND error_count < 3"
                    if start_id:
                        where_clause += f" AND id >= {start_id}"
                    
                    rows = await conn.fetch(
                        f"""
                        SELECT id, cast_id, text
                        FROM public.test_casts
                        {where_clause}
                        ORDER BY id
                        LIMIT $1
                        """,
                        fetch_size
                    )
                else:
                    where_clause = "WHERE embedding_updated_at IS NULL AND text IS NOT NULL"
                    if start_id:
                        where_clause += f" AND id >= {start_id}"
                    
                    rows = await conn.fetch(
                        f"""
                        SELECT id, text
                        FROM public.casts
                        {where_clause}
                        ORDER BY id
                        LIMIT $1
                        """,
                        fetch_size
                    )
            
            if not rows:
                break
                
            # Split into exact batch_size chunks for parallel processing
            batches = []
            for i in range(0, len(rows), self.batch_size):
                batch = rows[i:i + self.batch_size]
                if len(batch) == self.batch_size:  # Only process full batches
                    if self.test_mode:
                        texts = [row['text'] for row in batch]
                        ids = [row['id'] for row in batch]
                        cast_ids = [row['cast_id'] for row in batch]
                        batches.append((texts, ids, cast_ids))
                    else:
                        texts = [row['text'] for row in batch]
                        ids = [row['id'] for row in batch]
                        batches.append((texts, ids, None))
            
            if not batches:
                continue
            
            # Process batches in parallel
            start_time = asyncio.get_event_loop().time()
            
            # Process in chunks to not overwhelm the GPU
            chunk_size = 4  # Process 4 batches at a time
            for i in range(0, len(batches), chunk_size):
                chunk = batches[i:i + chunk_size]
                await asyncio.gather(*[
                    self._process_batch(texts, ids, cast_ids)
                    for texts, ids, cast_ids in chunk
                ])
            
            # Calculate throughput and adjust instances
            duration = asyncio.get_event_loop().time() - start_time
            throughput = len(rows) / duration
            self._adjust_instances(throughput)
            
            logger.info(f"Processed {len(rows)} casts at {throughput:.0f} texts/second")