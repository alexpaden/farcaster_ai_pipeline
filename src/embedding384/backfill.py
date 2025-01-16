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
NUM_PROCESSES = 48  # Number of parallel processes
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
        
        sample_texts = ["This is a longer initialization text that will ensure adequate buffer sizes"] * 32
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
        
        self.input_buffers = {
            'input_ids': torch.zeros((self.batch_size, MAX_SEQ_LENGTH), dtype=torch.long, device=self.device),
            'attention_mask': torch.zeros((self.batch_size, MAX_SEQ_LENGTH), dtype=torch.long, device=self.device)
        }
        
        with torch.inference_mode():
            outputs = self.model(inputs['input_ids'], inputs['attention_mask'])
            embeddings = self._mean_pooling(outputs, inputs['attention_mask'])
            self.output_buffer = torch.zeros(
                (self.batch_size, embeddings.shape[1]),
                dtype=torch.float16,
                device=self.device
            )
        
        del base_model, traced_model, outputs, embeddings
        gc.collect()
        
        # Warm up silently
        warmup_texts = ["Warm-up sentence"] * 8
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
            )
            
            for k, v in inputs.items():
                if k in self.input_buffers:
                    self.input_buffers[k][:v.size(0), :v.size(1)] = v.to(self.device)
                    inputs[k] = self.input_buffers[k][:v.size(0), :v.size(1)]
            
            outputs = self.model(inputs['input_ids'], inputs['attention_mask'])
            embeddings = self._mean_pooling(outputs, inputs['attention_mask'])
            
            self.output_buffer[:embeddings.size(0)] = embeddings
            return self.output_buffer[:embeddings.size(0)].cpu().numpy()

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

async def process_batch(conn, ids: List[int]) -> List[Dict[str, Any]]:
    """Fetch a batch of casts by IDs."""
    if not ids:
        return []
    
    rows = await conn.fetch("""
        SELECT id, text
        FROM public.casts
        WHERE id = ANY($1)
        AND embedding384 IS NULL
        ORDER BY id
    """, ids)
    
    return [dict(row) for row in rows]

async def update_embeddings(conn, id_embedding_pairs: List[Tuple[int, List[float]]]):
    """Update embeddings in batches."""
    if not id_embedding_pairs:
        return
    
    await conn.executemany("""
        UPDATE public.casts
        SET embedding384 = $2
        WHERE id = $1
        AND embedding384 IS NULL
    """, [(id, embedding.tolist()) for id, embedding in id_embedding_pairs])

async def process_range(start_id: int, end_id: int, batch_size: int, progress_queue=None):
    """Process a range of IDs in batches."""
    model = OptimizedEmbeddingModel(batch_size)
    
    try:
        async with db.pool.acquire() as conn:
            current_id = start_id
            while current_id <= end_id:
                # Get next batch of IDs
                id_batch = list(range(current_id, min(current_id + batch_size, end_id + 1)))
                
                # Process batch
                rows = await process_batch(conn, id_batch)
                if not rows:
                    current_id += batch_size
                    continue
                
                # Generate embeddings
                texts = [row['text'] for row in rows]
                embeddings = model.encode(texts)
                
                # Update database
                id_embedding_pairs = list(zip([row['id'] for row in rows], embeddings))
                await update_embeddings(conn, id_embedding_pairs)
                
                # Report progress
                if progress_queue:
                    progress_queue.put(len(rows))
                
                current_id += batch_size
                
    except Exception as e:
        print(f"Error processing range {start_id}-{end_id}: {str(e)}")
        raise

async def main():
    """Main backfill orchestration."""
    try:
        start_time = time.time()
        print("\nStarting embedding backfill...")
        
        # Initialize database and run migrations
        await db.initialize_pool()
        await db.run_migrations(module_path=str(Path(__file__).parent))
        
        # Get table statistics
        total_rows, min_id, max_id = await get_table_stats(db.pool)
        if max_id == 0:
            print("No rows to process")
            return
        
        print(f"\nProcessing casts table:")
        print(f"Estimated total rows: {total_rows:,}")
        print(f"Processing ID range: {min_id:,} - {max_id:,}")
        
        # Calculate ranges for each process
        id_range = max_id - min_id + 1
        chunk_size = id_range // NUM_PROCESSES
        ranges = [
            (min_id + i * chunk_size, min_id + (i + 1) * chunk_size - 1)
            for i in range(NUM_PROCESSES)
        ]
        ranges[-1] = (ranges[-1][0], max_id)  # Ensure last range includes max_id
        
        # Set up progress tracking
        progress_queue = multiprocessing.Queue()
        processes = []
        
        # Start worker processes
        print(f"\nStarting {NUM_PROCESSES} worker processes...")
        for start, end in ranges:
            p = multiprocessing.Process(
                target=asyncio.run,
                args=(process_range(start, end, BATCH_SIZE, progress_queue),)
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
