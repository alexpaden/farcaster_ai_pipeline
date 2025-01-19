"""
Batch process embedding generation for casts table.
Uses async prefetching and parallel processing with TorchScript optimization and float16 precision on MPS.
Stores embeddings as int8 vectors for efficiency.
"""

# Configuration settings
BATCH_SIZE = 256  # Optimal batch size for MPS
NUM_PROCESSES = 1  # Number of parallel processes
PREFETCH_BATCHES = 3  # Increased from 3 to keep pipeline fuller
BATCH_SIZE_ROWS = 100000  # Number of rows to process per instance
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_SEQ_LENGTH = 256  # Maximum sequence length for tokenization (model max is 256)
DB_TIMEOUT = 60  # Database timeout in seconds

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
    
    def get_recent_tps(self) -> float:
        """Calculate recent transactions per second."""
        current_time = time.time()
        time_since_last = current_time - self.last_log_time
        if time_since_last > 0:
            return (self.total_processed - self.last_processed) / time_since_last
        return 0.0
    
    def get_eta_minutes(self) -> float:
        """Calculate estimated time remaining in minutes."""
        current_time = time.time()
        total_time = current_time - self.start_time
        if self.total_processed > 0 and total_time > 0:
            overall_tps = self.total_processed / total_time
            if overall_tps > 0:
                return (self.total_remaining / overall_tps) / 60
        return 0.0
    
    def update(self, processed: int):
        """Update stats with newly processed rows."""
        self.total_processed += processed
        self.total_remaining = max(0, self.total_remaining - processed)
        
        # Log every 5 seconds
        current_time = time.time()
        time_since_last = current_time - self.last_log_time
        if time_since_last >= 5:
            total_time = current_time - self.start_time
            recent_tps = self.get_recent_tps()
            overall_tps = self.total_processed / total_time if total_time > 0 else 0
            
            print(f"\nProgress update:")
            print(f"Recent TPS: {recent_tps:,.1f}")
            print(f"Overall TPS: {overall_tps:,.1f}")
            print(f"Processed: {self.total_processed:,} rows")
            print(f"Remaining: {self.total_remaining:,} rows")
            if self.total_remaining > 0 and overall_tps > 0:
                print(f"ETA: {self.get_eta_minutes():.1f} minutes")
            
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
    """Get current count of unprocessed rows using optimized function."""
    async with pool.acquire() as conn:
        await conn.execute("SET statement_timeout = '60s'")
        if last_id == 0:
            # Use optimized count function for full table count
            result = await conn.fetchval("SELECT get_unprocessed_count()")
        else:
            # For partial counts, use regular query since we need id filter
            result = await conn.fetchval("""
                SELECT count(*)
                FROM public.casts
                WHERE embedding384 IS NULL 
                AND id > $1
                AND text IS NOT NULL 
                AND length(trim(text)) > 0
            """, last_id)
        return int(result or 0)

@dataclass
class ProcessingMetrics:
    """Track detailed processing metrics."""
    model_inference_times: List[float] = field(default_factory=list)
    db_update_times: List[float] = field(default_factory=list)
    batch_sizes: List[int] = field(default_factory=list)
    memory_usage: List[float] = field(default_factory=list)
    fetch_times: List[float] = field(default_factory=list)
    quantize_times: List[float] = field(default_factory=list)
    upsert_times: List[float] = field(default_factory=list)
    
    def add_timing(self, category: str, duration: float):
        if category == 'model_inference':
            self.model_inference_times.append(duration)
        elif category == 'db_update':
            self.db_update_times.append(duration)
        elif category == 'fetch':
            self.fetch_times.append(duration)
        elif category == 'quantize':
            self.quantize_times.append(duration)
        elif category == 'upsert':
            self.upsert_times.append(duration)
    
    def add_batch_size(self, size: int):
        self.batch_sizes.append(size)
    
    def add_memory(self, cpu_mem: float, gpu_mem: float):
        self.memory_usage.append(cpu_mem)
    
    def log_stats(self):
        """Log current statistics."""
        if len(self.model_inference_times) % 10 == 0:  # Increased frequency
            def safe_avg(lst): return sum(lst[-100:]) / len(lst[-100:]) if lst else 0
            def safe_p95(lst): return sorted(lst[-100:])[int(len(lst[-100:])*0.95)] if len(lst) >= 100 else 0
            
            print("\nPipeline Performance:")
            print(f"Fetch Time: avg={safe_avg(self.fetch_times):.3f}s, p95={safe_p95(self.fetch_times):.3f}s")
            print(f"Model Inference: avg={safe_avg(self.model_inference_times):.3f}s, p95={safe_p95(self.model_inference_times):.3f}s")
            print(f"Quantization: avg={safe_avg(self.quantize_times):.3f}s, p95={safe_p95(self.quantize_times):.3f}s")
            print(f"DB Upsert: avg={safe_avg(self.upsert_times):.3f}s, p95={safe_p95(self.upsert_times):.3f}s")
            print(f"Memory Usage (MB): current={self.memory_usage[-1]:.1f}")

class BatchProcessor:
    def __init__(self, pool, batch_size: int):
        self.pool = pool
        self.batch_size = batch_size
        self.prefetch_queue = asyncio.Queue(maxsize=PREFETCH_BATCHES)
        self.last_id = 0
        self.done = False
        self._prefetch_task = None
        self.metrics = ProcessingMetrics()

    async def start(self):
        """Start the prefetch worker."""
        self._prefetch_task = asyncio.create_task(self.prefetch_worker())

    async def prefetch_worker(self):
        """Continuously prefetch next batches."""
        try:
            empty_count = 0
            while not self.done:
                current_size = self.prefetch_queue.qsize()
                
                if current_size < PREFETCH_BATCHES:
                    batch, fetch_time = await fetch_next_batch(self.pool, self.last_id, self.batch_size)
                    
                    if not batch:
                        empty_count += 1
                        if empty_count == 1:  # Log on first empty
                            print(f"\nPrefetch queue empty (size: {current_size}/{PREFETCH_BATCHES})")
                        
                        # If queue completely empty, wait longer
                        if current_size == 0:
                            await asyncio.sleep(5)  # Wait 5s when queue empty
                        else:
                            await asyncio.sleep(0.5)  # Wait 0.5s when partially full
                            
                        continue
                    
                    empty_count = 0
                    self.last_id = batch[-1]['id']
                    await self.prefetch_queue.put((batch, fetch_time))
                    
                    if current_size == 0:
                        print(f"Prefetch: Refilled empty queue with {len(batch)} rows")
                
                await asyncio.sleep(0.1)
                
        except Exception as e:
            print(f"Error in prefetch worker: {str(e)}")
            self.done = True
            raise

    async def get_next_batch(self) -> Optional[Tuple[List[Dict[str, Any]], float]]:
        """Get next batch from prefetch queue."""
        if self.done and self.prefetch_queue.empty():
            return None
        try:
            return await self.prefetch_queue.get()
        except asyncio.QueueEmpty:
            return None

async def fetch_next_batch(pool, last_id: int = 0, limit: int = 100) -> Tuple[List[Dict[str, Any]], float]:
    """Fetch next batch of unprocessed casts."""
    fetch_start = time.time()
    async with pool.acquire() as conn:
        await conn.execute("SET statement_timeout = '60s'")
        
        # Simplified query to reduce lock contention
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
        
        if not rows:
            return [], time.time() - fetch_start
            
        # Mark rows as in progress
        ids = [row['id'] for row in rows]
        await conn.execute("""
            UPDATE public.casts
            SET embedding384_updated_at = NOW()
            WHERE id = ANY($1::bigint[])
        """, ids)
        
        fetch_time = time.time() - fetch_start
        return [dict(row) for row in rows], fetch_time

class EmbeddingProcessor:
    """Handles model initialization and inference."""
    def __init__(self, batch_size: int, quiet: bool = True):
        self.model = OptimizedEmbeddingModel(batch_size, quiet=quiet)
        self.batch_size = batch_size
    
    def process_batch(self, texts: List[str], ids: List[int]) -> Tuple[np.ndarray, List[int], float]:
        """Process a batch of texts."""
        all_embeddings = []
        all_batch_ids = []
        all_sims = []
        
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_ids = ids[i:i + self.batch_size]
            
            embeddings, sim = self.model.encode(batch_texts)
            all_embeddings.append(embeddings)
            all_batch_ids.extend(batch_ids)
            all_sims.append(sim)
        
        # Combine all batches
        combined_embeddings = np.vstack(all_embeddings)
        avg_sim = sum(all_sims) / len(all_sims) if all_sims else 0
        
        return combined_embeddings, all_batch_ids, avg_sim

def run_processor_instance(texts: List[str], ids: List[int], batch_size: int, progress_queue=None, metrics_queue=None):
    """Run processing in a separate process."""
    metrics = ProcessingMetrics()
    start_time = time.time()
    pid = os.getpid()
    
    try:
        # Initialize model once
        processor = EmbeddingProcessor(batch_size, quiet=True)
        
        # Track memory before inference
        cpu_mem = get_process_memory(pid)
        gpu_mem = get_gpu_memory()
        metrics.add_memory(cpu_mem, gpu_mem)
        
        # Time model inference
        inference_start = time.time()
        embeddings, batch_ids, sim = processor.process_batch(texts, ids)
        metrics.add_timing('model_inference', time.time() - inference_start)
        
        # Signal progress
        if progress_queue:
            progress_queue.put(len(texts))
        
        if metrics_queue:
            metrics_queue.put(metrics)
        
        return embeddings, batch_ids, sim
            
    except Exception as e:
        print(f"Error in processor instance: {str(e)}")
        raise

@dataclass
class ProcessInfo:
    """Track process health and work status."""
    pid: int
    start_time: float
    last_active: float
    rows_processed: int = 0
    is_alive: bool = True
    
    def update_activity(self):
        self.last_active = time.time()
    
    def is_stale(self) -> bool:
        return time.time() - self.last_active > 60  # Process considered stale after 1 minute

class ProcessManager:
    def __init__(self, num_processes: int):
        self.processes: Dict[int, ProcessInfo] = {}
        self.num_processes = num_processes
        self.lock = asyncio.Lock()
    
    async def register_process(self, pid: int):
        async with self.lock:
            now = time.time()
            self.processes[pid] = ProcessInfo(pid=pid, start_time=now, last_active=now)
            print(f"Process {pid} registered")
    
    async def update_process(self, pid: int, rows_processed: int):
        async with self.lock:
            if pid in self.processes:
                self.processes[pid].rows_processed += rows_processed
                self.processes[pid].update_activity()
    
    async def check_health(self) -> List[int]:
        """Return list of stale process PIDs."""
        async with self.lock:
            stale_pids = []
            for pid, info in self.processes.items():
                try:
                    process = psutil.Process(pid)
                    if not process.is_running() or info.is_stale():
                        info.is_alive = False
                        stale_pids.append(pid)
                        print(f"\nProcess {pid} appears stale or dead (last active: {time.time() - info.last_active:.0f}s ago)")
                except psutil.NoSuchProcess:
                    info.is_alive = False
                    stale_pids.append(pid)
                    print(f"\nProcess {pid} has died")
            return stale_pids

async def process_casts(pool, batch_size: int, num_instances: int):
    """Process casts table with specified batch size and number of instances."""
    print(f"\nProcessing casts with batch_size={batch_size}, instances={num_instances}")
    
    stats = ProcessingStats()
    metrics = ProcessingMetrics()
    stats.total_remaining = await get_unprocessed_estimate(pool)
    print(f"Starting processing of {stats.total_remaining:,} rows...")
    
    processor = BatchProcessor(pool, BATCH_SIZE_ROWS)
    process_manager = ProcessManager(num_instances)
    await processor.start()
    
    # Initialize model in main process
    embedding_processor = EmbeddingProcessor(batch_size, quiet=True) if num_instances == 1 else None
    last_batch_end = time.time()  # Track time between batches
    
    async def health_monitor():
        while not processor.done:
            stale_pids = await process_manager.check_health()
            if stale_pids:
                print("\nRestarting stale processes...")
                async with pool.acquire() as conn:
                    await conn.execute("""
                        UPDATE casts
                        SET embedding384_updated_at = NULL
                        WHERE embedding384 IS NULL 
                        AND embedding384_updated_at IS NOT NULL
                        AND embedding384_updated_at < NOW() - INTERVAL '5 minutes'
                    """)
            
            async with pool.acquire() as conn:
                await conn.execute("SET statement_timeout = '60s'")
                verify_count = await conn.fetchval("""
                    SELECT count(*) 
                    FROM public.casts 
                    WHERE embedding384 IS NOT NULL 
                    AND text IS NOT NULL 
                    AND length(trim(text)) > 0
                """)
                unprocessed_count = await conn.fetchval("SELECT get_unprocessed_count()")
                print(f"\nVerification - Processed rows in DB: {verify_count:,}")
                print(f"Verification - Unprocessed rows in DB: {unprocessed_count:,}")
            
            await asyncio.sleep(30)
    
    health_task = asyncio.create_task(health_monitor())
    
    try:
        while True:
            batch_start = time.time()
            gap_time = batch_start - last_batch_end
            if gap_time > 1.0:  # Log gaps longer than 1 second
                print(f"\nGap between batches: {gap_time:.1f}s")
            
            fetch_start = time.time()
            batch, fetch_time = await processor.get_next_batch()
            metrics.add_timing('fetch', time.time() - fetch_start)
            
            if not batch:
                remaining = await get_unprocessed_estimate(pool, processor.last_id)
                if remaining == 0:
                    break
                print(f"\nNo batch available but {remaining:,} rows remain (waiting for in-progress work)")
                await asyncio.sleep(1)
                continue
            
            texts = [cast['text'] for cast in batch]
            ids = [cast['id'] for cast in batch]
            
            if num_instances == 1:
                # Process in current process using reused model
                inference_start = time.time()
                embeddings, batch_ids, sim = embedding_processor.process_batch(texts, ids)
                inference_time = time.time() - inference_start
                metrics.add_timing('model_inference', inference_time)
                
                # Update database with embeddings in larger chunks
                chunk_size = 5000
                for i in range(0, len(batch_ids), chunk_size):
                    chunk_start = time.time()
                    chunk_ids = batch_ids[i:i + chunk_size]
                    chunk_embeddings = embeddings[i:i + chunk_size]
                    
                    upsert_start = time.time()
                    async with pool.acquire() as conn:
                        async with conn.transaction():
                            value_strings = [
                                f"({id_}, '[{','.join(map(str, emb))}]')"
                                for id_, emb in zip(chunk_ids, chunk_embeddings)
                            ]
                            
                            values_clause = ','.join(value_strings)
                            update_sql = f"""
                                UPDATE casts AS t
                                SET 
                                    embedding384 = v.embedding::vector,
                                    embedding384_updated_at = NOW()
                                FROM (VALUES {values_clause}) AS v(id, embedding)
                                WHERE t.id = v.id
                            """
                            result = await conn.execute(update_sql)
                            
                            if hasattr(result, 'split'):
                                updated = int(result.split()[-1])
                                if updated != len(chunk_ids):
                                    print(f"\nWarning: Expected to update {len(chunk_ids)} rows but updated {updated}")
                    
                    upsert_time = time.time() - upsert_start
                    metrics.add_timing('upsert', upsert_time)
                    stats.update(len(chunk_ids))
                    
                    # Get current memory usage
                    cpu_mem = get_process_memory(os.getpid())
                    gpu_mem = get_gpu_memory()
                    metrics.add_memory(cpu_mem, gpu_mem)
                    
                    chunk_time = time.time() - chunk_start
                    
                    # Log performance metrics with each chunk
                    print("\nPipeline Performance:")
                    print(f"Fetch Time: avg={sum(metrics.fetch_times[-100:])/len(metrics.fetch_times[-100:]):.3f}s")
                    print(f"Model Inference: avg={sum(metrics.model_inference_times[-100:])/len(metrics.model_inference_times[-100:]):.3f}s")
                    print(f"DB Upsert: avg={sum(metrics.upsert_times[-100:])/len(metrics.upsert_times[-100:]):.3f}s")
                    print(f"Memory Usage (MB): CPU={cpu_mem:.1f}, GPU={gpu_mem:.1f}")
                    print(f"Chunk Processing Time: {chunk_time:.3f}s")
                
                last_batch_end = time.time()
                batch_time = last_batch_end - batch_start
                print(f"Total Batch Time: {batch_time:.3f}s")
            else:
                # Multi-process handling remains the same
                texts_per_instance = len(texts) // num_instances
                progress_queue = multiprocessing.Queue()
                metrics_queue = multiprocessing.Queue()
                processes = []
                
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
                    await process_manager.register_process(p.pid)
                    processes.append(p)
                
                # Monitor progress
                completed = 0
                while completed < len(processes):
                    try:
                        progress = progress_queue.get(timeout=1)
                        completed += 1
                        stats.update(texts_per_instance)
                        await process_manager.update_process(processes[completed-1].pid, texts_per_instance)
                    except:
                        continue
                
                for p in processes:
                    p.join()
    
    finally:
        processor.done = True
        health_task.cancel()
        try:
            await health_task
        except asyncio.CancelledError:
            pass
        
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