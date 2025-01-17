"""
Backfill embeddings for all casts where embedding384 is null.
Uses optimized model with parallel processing for high throughput.
"""

import os
import warnings
import sys
import psutil
import time
import json
from pathlib import Path
import subprocess
import asyncio
from typing import List, Dict, Any, Tuple
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm
import torch
import torch.nn as nn
import gc
import multiprocessing
from transformers import AutoTokenizer, AutoModel

from src.db.connect import db

# Configuration
BATCH_SIZE = 256  # Optimal batch size for MPS
NUM_PROCESSES = 12  # Number of parallel processes to match benchmark
MAX_SEQ_LENGTH = 256  # Maximum sequence length for tokenization
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Load environment variables and configure
load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", message=".*_is_quantized_training_enabled.*")
warnings.filterwarnings("ignore", message=".*loss_type=None.*")

class OptimizedEmbeddingModel:
    def __init__(self, batch_size: int, quiet: bool = True):
        self.device = torch.device("mps")
        self.batch_size = batch_size
        self.model_name = MODEL_NAME
        self.quiet = quiet
        
        if not quiet:
            print(f"\nInitializing model with batch_size={batch_size} on MPS...")
        self._initialize_model()
        
    def _initialize_model(self):
        torch.set_num_threads(multiprocessing.cpu_count())
        gc.collect()
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        base_model = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        base_model.to(self.device)
        base_model.eval()
        
        # Initialize with full batch size for optimal performance
        sample_texts = ["This is a longer initialization text that will ensure adequate buffer sizes"] * self.batch_size
        inputs = self.tokenizer(
            sample_texts,
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.inference_mode():
            traced_model = torch.jit.trace(
                base_model,
                (inputs['input_ids'], inputs['attention_mask']),
                strict=False
            )
            self.model = torch.jit.optimize_for_inference(traced_model)
        
        del base_model, traced_model
        gc.collect()
        
        # Warm up with full batch
        warmup_texts = ["Warm-up sentence"] * self.batch_size
        _ = self.encode(warmup_texts)
        
    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output['last_hidden_state']
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        normalized = sum_embeddings / sum_mask
        return torch.nn.functional.normalize(normalized, p=2, dim=1)
    
    def encode(self, texts: List[str]) -> np.ndarray:
        with torch.inference_mode():
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=MAX_SEQ_LENGTH,
                return_tensors="pt"
            ).to(self.device)
            
            outputs = self.model(inputs['input_ids'], inputs['attention_mask'])
            embeddings = self._mean_pooling(outputs, inputs['attention_mask'])
            return embeddings.cpu().numpy()

async def get_table_stats(pool) -> Tuple[int, int, int]:
    """Get fast row estimate and ID range using pg_class statistics."""
    async with pool.acquire() as conn:
        # Get row estimate
        estimate = await conn.fetchval("""
            SELECT reltuples::bigint AS estimate
            FROM pg_class
            WHERE relname = 'casts'
        """)
        
        # Get ID range for null embeddings
        range_data = await conn.fetch("""
            SELECT 
                MIN(id) as min_id,
                MAX(id) as max_id
            FROM public.casts
            WHERE embedding384 IS NULL
        """)
        
        min_id = range_data[0]['min_id'] or 0
        max_id = range_data[0]['max_id'] or 0
        
        return int(estimate or 0), min_id, max_id

async def process_batch(conn, batch_size: int, max_rows: int = None) -> List[Dict[str, Any]]:
    """Fetch a batch of casts that need processing."""
    rows = await conn.fetch("""
        SELECT id, text 
        FROM public.casts 
        WHERE embedding384 IS NULL
        LIMIT $1
        FOR UPDATE SKIP LOCKED
    """, batch_size)
    
    return [dict(row) for row in rows]

async def update_embeddings(conn, id_embedding_pairs: List[Tuple[int, List[float]]]):
    """Update embeddings in batches."""
    if not id_embedding_pairs:
        return
    
    # Prepare value strings for batch update
    value_strings = []
    for id_, emb in id_embedding_pairs:
        vec_str = '[' + ','.join(map(str, emb.tolist())) + ']'
        value_strings.append(f"({id_}, '{vec_str}')")
    
    values_clause = ','.join(value_strings)
    
    # Single UPDATE statement for entire batch
    update_sql = f"""
        UPDATE casts AS t
        SET embedding384 = v.embedding::vector
        FROM (VALUES {values_clause}) AS v(id, embedding)
        WHERE t.id = v.id
    """
    
    await conn.execute(update_sql)

def process_range_sync(worker_id: int, batch_size: int, max_rows: int, progress_queue=None):
    """Synchronous wrapper for processing batches."""
    model = OptimizedEmbeddingModel(batch_size)
    processed_count = 0
    total_fetch_time = 0
    total_inference_time = 0
    total_update_time = 0
    batch_count = 0
    start_time = time.time()
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def process_range_async():
            nonlocal processed_count, total_fetch_time, total_inference_time, total_update_time, batch_count
            await db.initialize_pool()
            
            try:
                async with db.pool.acquire() as conn:
                    while processed_count < max_rows:
                        async with conn.transaction():
                            # Time DB fetch
                            fetch_start = time.time()
                            rows = await process_batch(conn, min(batch_size, max_rows - processed_count))
                            fetch_time = time.time() - fetch_start
                            total_fetch_time += fetch_time
                            
                            if not rows:
                                break
                            
                            # Time inference
                            inference_start = time.time()
                            texts = [row['text'] for row in rows]
                            embeddings = model.encode(texts)
                            inference_time = time.time() - inference_start
                            total_inference_time += inference_time
                            
                            # Time DB update
                            update_start = time.time()
                            id_embedding_pairs = list(zip([row['id'] for row in rows], embeddings))
                            await update_embeddings(conn, id_embedding_pairs)
                            update_time = time.time() - update_start
                            total_update_time += update_time
                            
                            rows_processed = len(rows)
                            processed_count += rows_processed
                            batch_count += 1
                            
                            # Log timing details every 10 batches
                            if batch_count % 10 == 0:
                                elapsed = time.time() - start_time
                                print(f"\nWorker {worker_id} stats after {batch_count} batches:")
                                print(f"Avg fetch time: {(total_fetch_time/batch_count)*1000:.1f}ms")
                                print(f"Avg inference time: {(total_inference_time/batch_count)*1000:.1f}ms")
                                print(f"Avg update time: {(total_update_time/batch_count)*1000:.1f}ms")
                                print(f"Overall TPS: {processed_count/elapsed:.1f}")
                            
                            if progress_queue:
                                progress_queue.put(rows_processed)
                            
                            # Force garbage collection periodically
                            if processed_count % (batch_size * 100) == 0:
                                gc.collect()
                        
            finally:
                await db.close_pool()
        
        loop.run_until_complete(process_range_async())
        loop.close()
        
    except Exception as e:
        print(f"Error in worker {worker_id}: {str(e)}")
        raise
    
    # Final timing summary
    elapsed = time.time() - start_time
    print(f"\nWorker {worker_id} final stats:")
    print(f"Total batches: {batch_count}")
    print(f"Avg fetch time: {(total_fetch_time/batch_count)*1000:.1f}ms")
    print(f"Avg inference time: {(total_inference_time/batch_count)*1000:.1f}ms")
    print(f"Avg update time: {(total_update_time/batch_count)*1000:.1f}ms")
    print(f"Overall TPS: {processed_count/elapsed:.1f}")
    
    return processed_count

async def main():
    """Main backfill orchestration."""
    try:
        start_time = time.time()
        print("\nStarting embedding backfill...")
        
        # Initialize database and run migrations
        await db.initialize_pool()
        await db.run_migrations(module_path=str(Path(__file__).parent))
        
        # Get table statistics
        total_rows = 100_000  # Fixed number of rows for benchmarking
        print(f"\nProcessing casts table:")
        print(f"Target rows: {total_rows:,}")
        
        # Set up progress tracking
        progress_queue = multiprocessing.Queue()
        processes = []
        
        # Calculate rows per worker
        rows_per_worker = total_rows // NUM_PROCESSES
        
        # Start worker processes
        print(f"\nStarting {NUM_PROCESSES} worker processes...")
        for worker_id in range(NUM_PROCESSES):
            worker_rows = rows_per_worker
            if worker_id == NUM_PROCESSES - 1:  # Last worker gets any remainder
                worker_rows = total_rows - (NUM_PROCESSES - 1) * rows_per_worker
                
            p = multiprocessing.Process(
                target=process_range_sync,
                args=(worker_id, BATCH_SIZE, worker_rows, progress_queue)
            )
            p.start()
            processes.append(p)
        
        # Monitor progress
        processed_rows = 0
        with tqdm(total=total_rows, desc="Processing", unit="rows") as pbar:
            while any(p.is_alive() for p in processes):
                try:
                    rows_processed = progress_queue.get(timeout=1)
                    processed_rows += rows_processed
                    pbar.update(rows_processed)
                except:
                    continue
        
        # Wait for completion
        for p in processes:
            p.join()
        
        # Print summary
        duration = time.time() - start_time
        print("\nBackfill complete!")
        print(f"Processed {processed_rows:,} rows in {duration:.1f}s")
        print(f"Average speed: {processed_rows/duration:.1f} rows/s")
        
    finally:
        await db.close_pool()

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    asyncio.run(main())
