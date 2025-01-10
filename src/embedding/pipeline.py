"""
Embedding pipeline for processing Farcaster casts using MiniLM-L6-v2.
Implements dynamic scaling and efficient batch processing.
"""

import os
import asyncio
import logging
from typing import Optional, List
import torch
from sentence_transformers import SentenceTransformer
import asyncpg
from pathlib import Path
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingPipeline:
    def __init__(self, test_mode: bool = False):
        self.test_mode = test_mode
        self.model = None
        self.pool = None
        self.batch_size = 256
        self.max_instances = 48
        self.current_instances = 1
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        
    async def initialize(self):
        """Initialize database connection and model."""
        # Load environment variables
        load_dotenv()
        db_host = os.getenv("DB_HOST", "localhost")
        db_name = os.getenv("DB_NAME", "postgres")
        db_user = os.getenv("DB_USER", "postgres")
        db_pass = os.getenv("DB_PASSWORD", "")
        db_port = int(os.getenv("DB_PORT", "5432"))
        
        # Initialize database pool
        self.pool = await asyncpg.create_pool(
            host=db_host,
            database=db_name,
            user=db_user,
            password=db_pass,
            port=db_port
        )
        
        # Initialize model
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.model.to(self.device)
        
        # Run migrations if needed
        await self._run_migrations()
        
    async def _run_migrations(self):
        """Run database migrations from SQL files."""
        async with self.pool.acquire() as conn:
            sql_dir = Path("sql")
            schema_file = sql_dir / "schema.sql"
            
            if schema_file.exists():
                with open(schema_file) as f:
                    await conn.execute(f.read())
    
    def _adjust_instances(self, throughput: float):
        """Dynamically adjust number of instances based on throughput."""
        target = min(int(throughput / 1000) + 1, self.max_instances)
        self.current_instances = max(1, min(target, self.max_instances))
        logger.info(f"Adjusted to {self.current_instances} instances")
    
    async def _process_batch(self, texts: List[str], cast_ids: List[int]) -> None:
        """Process a batch of texts and update database."""
        # Generate embeddings
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_tensor=True,
            device=self.device
        )
        
        # Convert to list for database storage
        embeddings_list = embeddings.cpu().numpy().tolist()
        
        # Update database
        async with self.pool.acquire() as conn:
            table = "test_casts" if self.test_mode else "casts"
            await conn.executemany(
                f"""
                UPDATE {table}
                SET embedding = $2::vector
                WHERE cast_id = $1
                """,
                zip(cast_ids, embeddings_list)
            )
    
    async def process_casts(self, start_id: Optional[int] = None):
        """Process all casts with dynamic scaling."""
        table = "test_casts" if self.test_mode else "casts"
        
        while True:
            # Fetch unprocessed casts
            async with self.pool.acquire() as conn:
                where_clause = "WHERE embedding IS NULL"
                if start_id:
                    where_clause += f" AND cast_id >= {start_id}"
                
                rows = await conn.fetch(
                    f"""
                    SELECT cast_id, text
                    FROM {table}
                    {where_clause}
                    LIMIT $1
                    """,
                    self.batch_size * self.current_instances
                )
            
            if not rows:
                break
                
            # Split into batches for parallel processing
            batches = []
            for i in range(0, len(rows), self.batch_size):
                batch = rows[i:i + self.batch_size]
                texts = [row['text'] for row in batch]
                cast_ids = [row['cast_id'] for row in batch]
                batches.append((texts, cast_ids))
            
            # Process batches in parallel
            start_time = asyncio.get_event_loop().time()
            await asyncio.gather(*[
                self._process_batch(texts, cast_ids)
                for texts, cast_ids in batches
            ])
            
            # Calculate throughput and adjust instances
            duration = asyncio.get_event_loop().time() - start_time
            throughput = len(rows) / duration
            self._adjust_instances(throughput)
            
            logger.info(f"Processed {len(rows)} casts at {throughput:.0f} texts/second")