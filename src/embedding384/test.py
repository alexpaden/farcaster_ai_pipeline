"""
Batch process embedding generation for casts table.
Uses async prefetching and parallel processing with TorchScript optimization and float16 precision on MPS.
Stores embeddings as int8 vectors for efficiency.
"""

# Configuration settings
BATCH_SIZE = 256  # Optimal batch size for MPS
NUM_PROCESSES = 18  # Number of parallel processes
PREFETCH_BATCHES = 2  # Number of batches to prefetch
BATCH_SIZE_ROWS = 100000  # Number of rows to process per instance
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_SEQ_LENGTH = 256  # Maximum sequence length for tokenization (model max is 256)
DB_TIMEOUT = 30  # Database timeout in seconds

import os
import warnings
import sys
import psutil
import time
import json
from pathlib import Path
import subprocess
import asyncio
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm
import torch
import torch.nn as nn
import gc
import multiprocessing
from transformers import AutoTokenizer, AutoModel

from src.db.connect import db

# Load environment variables
load_dotenv()

# Set environment variables and configure warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", message=".*_is_quantized_training_enabled.*")
warnings.filterwarnings("ignore", message=".*loss_type=None.*")

class OptimizedEmbeddingModel:
    def __init__(self, batch_size: int, quiet: bool = False):
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
    
    def encode(self, texts: List[str]) -> Tuple[np.ndarray, float]:
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
            float16_embeddings = self.output_buffer[:embeddings.size(0)]
            
            # Convert to CPU and float32 for quantization
            emb_cpu = float16_embeddings.detach().cpu().float()
            
            # Global scaling factor based on maximum absolute value across all vectors
            global_max_abs = emb_cpu.abs().max()
            global_scale = global_max_abs / 127.0
            
            # Quantize to int8 (-128 to 127) using global scale
            quantized = (emb_cpu / global_scale).round().clamp(-128, 127).to(torch.int8)
            
            # For similarity comparison:
            # 1. Convert original float16 to float32 for comparison
            orig_float32 = float16_embeddings.cpu().float()
            # 2. Convert quantized back to same scale as original
            quantized_float32 = (quantized.float() * global_scale)
            
            # Compute cosine similarity between original and quantized
            sim = cosine_similarity(orig_float32, quantized_float32)
            
            return quantized.numpy(), sim

def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute cosine similarity between two tensors."""
    return float(torch.nn.functional.cosine_similarity(a, b, dim=1).mean().item())

@dataclass
class BenchmarkMetrics:
    texts_per_second: float
    memory_usage_mb: float
    batch_size: int
    total_texts: int
    duration_seconds: float
    num_instances: int
    avg_cosine_sim: float = 0.0  # Average cosine similarity between float16 and int8

@dataclass
class ProcessingStats:
    """Track processing statistics across instances."""
    start_time: float = field(default_factory=time.time)
    total_processed: int = 0
    total_remaining: int = 0
    last_log_time: float = field(default_factory=time.time)
    last_processed: int = 0
    
    def update(self, processed: int):
        """Update stats with newly processed rows."""
        self.total_processed += processed
        self.total_remaining = max(0, self.total_remaining - processed)
        
        # Log every 5 seconds
        current_time = time.time()
        time_since_last = current_time - self.last_log_time
        if time_since_last >= 5:
            total_time = current_time - self.start_time
            recent_tps = (self.total_processed - self.last_processed) / time_since_last
            overall_tps = self.total_processed / total_time
            
            print(f"\nProgress update:")
            print(f"Recent TPS: {recent_tps:,.1f}")
            print(f"Overall TPS: {overall_tps:,.1f}")
            print(f"Processed: {self.total_processed:,} rows")
            print(f"Remaining: {self.total_remaining:,} rows")
            if self.total_remaining > 0 and overall_tps > 0:
                eta_seconds = self.total_remaining / overall_tps
                eta_minutes = eta_seconds / 60
                print(f"ETA: {eta_minutes:.1f} minutes")
            
            self.last_log_time = current_time
            self.last_processed = self.total_processed

def get_process_memory(pid):
    """Get memory usage for a process and all its children."""
    try:
        parent = psutil.Process(pid)
        memory = parent.memory_info().rss
        for child in parent.children(recursive=True):
            try:
                memory += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return memory / (1024 * 1024)  # Convert to MB
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0

def get_gpu_memory():
    """Get MPS memory usage."""
    try:
        result = subprocess.run(['ps', '-o', 'rss=', '-p', str(os.getpid())], 
                              capture_output=True, text=True)
        rss = int(result.stdout.strip()) / 1024  # Convert KB to MB
        return rss
    except:
        return 0

async def get_unprocessed_estimate(pool, last_id: int = 0) -> int:
    """Get current count of unprocessed rows."""
    async with pool.acquire() as conn:
        await conn.execute("SET statement_timeout = '30s'")
        result = await conn.fetchval("""
            SELECT (reltuples::float8 * 
                   (SELECT count(*)::float8 / count(*)::float8 
                    FROM public.casts TABLESAMPLE SYSTEM (0.1)
                    WHERE embedding384 IS NULL 
                    AND id > $1
                    AND text IS NOT NULL 
                    AND length(trim(text)) > 0))::bigint AS estimate
            FROM pg_class 
            WHERE relname = 'casts'
        """, last_id)
        return int(result or 0)

async def fetch_next_batch(pool, last_id: int = 0, limit: int = 100) -> List[Dict[str, Any]]:
    """Fetch next batch of unprocessed casts."""
    async with pool.acquire() as conn:
        await conn.execute("SET statement_timeout = '30s'")
        rows = await conn.fetch("""
            SELECT id, text 
            FROM public.casts 
            WHERE embedding384 IS NULL
            AND id > $1
            AND text IS NOT NULL 
            AND length(trim(text)) > 0
            ORDER BY id
            LIMIT $2
        """, last_id, limit)
        return [dict(row) for row in rows]

class BatchProcessor:
    def __init__(self, pool, batch_size: int):
        self.pool = pool
        self.batch_size = batch_size
        self.prefetch_queue = asyncio.Queue(maxsize=PREFETCH_BATCHES)
        self.last_id = 0
        self.done = False
        self._prefetch_task = None

    async def start(self):
        """Start the prefetch worker."""
        self._prefetch_task = asyncio.create_task(self.prefetch_worker())

    async def prefetch_worker(self):
        """Continuously prefetch next batches."""
        try:
            while not self.done:
                # Always try to keep the queue full
                while not self.done and self.prefetch_queue.qsize() < PREFETCH_BATCHES:
                    batch = await fetch_next_batch(self.pool, self.last_id, self.batch_size)
                    if not batch:
                        self.done = True
                        break
                    self.last_id = batch[-1]['id']
                    await self.prefetch_queue.put(batch)
                await asyncio.sleep(0.1)  # Small delay to prevent tight loop
        except Exception as e:
            print(f"Error in prefetch worker: {str(e)}")
            self.done = True
            raise

    async def get_next_batch(self) -> Optional[List[Dict[str, Any]]]:
        """Get next batch from prefetch queue."""
        if self.done and self.prefetch_queue.empty():
            return None
        try:
            return await self.prefetch_queue.get()
        except asyncio.QueueEmpty:
            return None

def run_processor_instance(texts: List[str], ids: List[int], batch_size: int, progress_queue=None, metrics_queue=None):
    """Run processing in a separate process."""
    model = OptimizedEmbeddingModel(batch_size, quiet=True)
    
    start_time = time.time()
    pid = os.getpid()
    peak_memory = 0
    total_sim = 0.0
    batch_count = 0
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(db.initialize_pool())
        
        async def process_batches():
            nonlocal total_sim, batch_count
            async with db.pool.acquire() as conn:
                await conn.execute("SET statement_timeout = '30s'")
                async with conn.transaction():
                    for i in range(0, len(texts), batch_size):
                        batch_texts = texts[i:i + batch_size]
                        batch_ids = ids[i:i + batch_size]
                        embeddings, sim = model.encode(batch_texts)
                        total_sim += sim
                        batch_count += 1
                        
                        # Prepare value strings for batch update
                        value_strings = []
                        for id_, emb in zip(batch_ids, embeddings):
                            vec_str = '[' + ','.join(map(str, emb)) + ']'
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
                        
                        nonlocal peak_memory
                        current_memory = get_process_memory(pid) + get_gpu_memory()
                        peak_memory = max(peak_memory, current_memory)
                        
                        if progress_queue:
                            progress_queue.put(1)
        
        loop.run_until_complete(process_batches())
        
        duration = time.time() - start_time
        avg_sim = total_sim / batch_count if batch_count > 0 else 0.0
        
        metrics = BenchmarkMetrics(
            texts_per_second=len(texts) / duration,
            memory_usage_mb=peak_memory,
            batch_size=batch_size,
            total_texts=len(texts),
            duration_seconds=duration,
            num_instances=1,
            avg_cosine_sim=avg_sim
        )
        
        if metrics_queue:
            metrics_queue.put(metrics)
            
        # Clean up
        loop.run_until_complete(db.close_pool())
        loop.close()
        
        return metrics
        
    except Exception as e:
        print(f"Error in processor instance: {str(e)}")
        raise

async def process_casts(pool, batch_size: int, num_instances: int):
    """Process casts table with specified batch size and number of instances."""
    print(f"\nProcessing casts with batch_size={batch_size}, instances={num_instances}")
    
    # Initialize stats
    stats = ProcessingStats()
    stats.total_remaining = await get_unprocessed_estimate(pool)
    print(f"Starting processing of {stats.total_remaining:,} rows...")
    
    processor = BatchProcessor(pool, BATCH_SIZE_ROWS)
    await processor.start()  # Start prefetching immediately
    
    try:
        while True:
            batch = await processor.get_next_batch()
            if not batch:
                # Double check if we're really done
                remaining = await get_unprocessed_estimate(pool, processor.last_id)
                if remaining == 0:
                    break
                print(f"\nNo batch available but {remaining:,} rows remaining, retrying...")
                await asyncio.sleep(1)
                continue
                
            texts = [cast['text'] for cast in batch]
            ids = [cast['id'] for cast in batch]
            
            if num_instances == 1:
                metrics = run_processor_instance(texts, ids, batch_size)
                stats.update(len(texts))
            else:
                texts_per_instance = len(texts) // num_instances
                progress_queue = multiprocessing.Queue()
                metrics_queue = multiprocessing.Queue()
                processes = []
                
                # Start processes
                for i in range(num_instances):
                    start_idx = i * texts_per_instance
                    end_idx = start_idx + texts_per_instance
                    instance_texts = texts[start_idx:end_idx]
                    instance_ids = ids[start_idx:end_idx]
                    
                    p = multiprocessing.Process(
                        target=run_processor_instance,
                        args=(instance_texts, instance_ids, batch_size, progress_queue, metrics_queue)
                    )
                    p.start()
                    processes.append(p)
                
                # Monitor progress
                completed_batches = 0
                rows_per_batch = batch_size * num_instances
                while any(p.is_alive() for p in processes):
                    try:
                        progress = progress_queue.get(timeout=1)
                        completed_batches += 1
                        stats.update(rows_per_batch)
                        
                        # Update remaining count periodically
                        if completed_batches % 10 == 0:
                            stats.total_remaining = await get_unprocessed_estimate(pool, processor.last_id)
                    except:
                        continue
                
                # Wait for completion
                for p in processes:
                    p.join()
                
                while not metrics_queue.empty():
                    _ = metrics_queue.get()
                
                gc.collect()
    
    finally:
        processor.done = True
        if processor._prefetch_task:
            await processor._prefetch_task
        
        # Final stats
        total_time = time.time() - stats.start_time
        print("\nProcessing complete!")
        print(f"Total processed: {stats.total_processed:,} rows")
        print(f"Overall TPS: {stats.total_processed/total_time:,.1f}")
        print(f"Total time: {total_time/60:.1f} minutes")

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
        
        await process_casts(db.pool, BATCH_SIZE, NUM_PROCESSES)
        
        total_duration = time.time() - start_time
        print(f"\nTotal processing time: {total_duration:.1f}s")
        
    finally:
        await db.close_pool()

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    asyncio.run(main()) 