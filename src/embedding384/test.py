"""
Batch process embedding generation for casts table.
Uses async prefetching and parallel processing with TorchScript optimization and float16 precision on MPS.
Stores embeddings as int8 vectors for efficiency.
"""

# Configuration settings
BATCH_SIZE = 2048  # Embedding batch size
BATCH_SIZE_ROWS = 200000  # Number of rows to process per instance
PREFETCH_BATCHES = 2  # Number of batches to prefetch
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_SEQ_LENGTH = 256  # Maximum sequence length for tokenization
DB_TIMEOUT = 60  # Database timeout in seconds
CHUNK_SIZE = 15000  # Rows to update in one transaction chunk

import os
import warnings
import sys
import psutil
import time
from pathlib import Path
import subprocess
import asyncio
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np
from dotenv import load_dotenv
import torch
import gc
import multiprocessing
from transformers import AutoTokenizer, AutoModel

from src.db.connect import db

# Load environment variables
load_dotenv()

@dataclass
class InferenceMetrics:
    """Track detailed inference metrics."""
    tokenization_time: float
    model_forward_time: float
    pooling_time: float
    quantization_time: float
    total_time: float
    batch_size: int
    texts_per_second: float

# Configure warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", message=".*_is_quantized_training_enabled.*")
warnings.filterwarnings("ignore", message=".*loss_type=None.*")

def quantize_worker(queue, shared_dict, shared_event):
    """Worker process for quantizing embeddings."""
    while True:
        item = queue.get()
        if item is None:  # Poison pill
            break
            
        embeddings = item
        # Quantize on CPU
        emb_cpu = embeddings.float()  # Already on CPU
        global_max_abs = emb_cpu.abs().max()
        global_scale = global_max_abs / 127.0
        quantized = (emb_cpu / global_scale).round().clamp(-128, 127).to(torch.int8)
        
        # Store results
        shared_dict['quantized'] = quantized.numpy()
        shared_dict['scale'] = global_scale.item()
        shared_event.set()

class OptimizedEmbeddingModel:
    def __init__(self, batch_size: int, quiet: bool = False):
        self.device = torch.device("mps")
        self.batch_size = batch_size
        self.quiet = quiet
        self.model_name = MODEL_NAME
        
        if not quiet:
            print(f"\nInitializing model with batch_size={batch_size} on MPS...")
        
        self._initialize_model()
        
    def _initialize_model(self):
        print("\nInitializing model...")
        torch.set_num_threads(multiprocessing.cpu_count())
        gc.collect()
        
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        print("Loading base model...")
        base_model = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        print("Moving model to MPS...")
        base_model.to(self.device)
        base_model.eval()
        
        print("Preparing sample inputs...")
        sample_texts = ["This is a longer initialization text that will ensure adequate buffer sizes"] * 32
        inputs = self.tokenizer(
            sample_texts,
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            return_tensors="pt"
        ).to(self.device)
        
        print("Tracing model...")
        with torch.inference_mode():
            traced_model = torch.jit.trace(
                base_model,
                (inputs['input_ids'], inputs['attention_mask']),
                strict=False
            )
            print("Optimizing traced model...")
            self.model = torch.jit.optimize_for_inference(traced_model)
        
        print("Initializing buffers...")
        self.input_buffers = {
            'input_ids': torch.zeros((self.batch_size, MAX_SEQ_LENGTH), dtype=torch.long, device=self.device),
            'attention_mask': torch.zeros((self.batch_size, MAX_SEQ_LENGTH), dtype=torch.long, device=self.device)
        }
        
        print("Running warmup inference...")
        with torch.inference_mode():
            outputs = self.model(inputs['input_ids'], inputs['attention_mask'])
            embeddings = self._mean_pooling(outputs, inputs['attention_mask'])
            self.output_buffer = torch.zeros(
                (self.batch_size, embeddings.shape[1]),
                dtype=torch.float16,
                device=self.device
            )
        
        print("Cleaning up...")
        del base_model, traced_model, outputs, embeddings
        gc.collect()
        
        print("Running warmup encode...")
        warmup_texts = ["Warm-up sentence"] * 8
        _ = self.encode(warmup_texts)
        
        print("Model initialization complete!")
        
    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output['last_hidden_state']
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        normalized = sum_embeddings / sum_mask
        return torch.nn.functional.normalize(normalized, p=2, dim=1)
    
    def encode(self, texts: List[str]) -> Tuple[np.ndarray, float, InferenceMetrics]:
        """Single-process inline quantization (no external worker)."""
        start_time = time.time()
        
        # 1. Tokenize
        tokenize_start = time.time()
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, max_length=MAX_SEQ_LENGTH, return_tensors="pt"
        )
        tokenize_time = time.time() - tokenize_start
        
        # Move to MPS
        for k, v in inputs.items():
            if k in self.input_buffers:
                self.input_buffers[k][:v.size(0), :v.size(1)] = v.to(self.device)
                inputs[k] = self.input_buffers[k][:v.size(0), :v.size(1)]
        
        # 2. Forward
        with torch.inference_mode():
            forward_start = time.time()
            outputs = self.model(inputs['input_ids'], inputs['attention_mask'])
            forward_time = time.time() - forward_start
            
            # 3. Pooling
            pool_start = time.time()
            embeddings = self._mean_pooling(outputs, inputs['attention_mask'])
            self.output_buffer[:embeddings.size(0)] = embeddings
            float16_embeddings = self.output_buffer[:embeddings.size(0)]
            pool_time = time.time() - pool_start
            
            # 4. Quantize inline
            quantize_start = time.time()
            emb_cpu = float16_embeddings.detach().cpu().float()
            global_max_abs = emb_cpu.abs().max()
            global_scale = global_max_abs / 127.0
            quantized_cpu = (emb_cpu / global_scale).round().clamp(-128, 127).to(torch.int8)
            sim = float(
                torch.nn.functional.cosine_similarity(
                    emb_cpu, (quantized_cpu.float() * global_scale), dim=1
                ).mean().item()
            )
            quantize_time = time.time() - quantize_start
        
        # 5. Metrics
        total_time = time.time() - start_time
        metrics = InferenceMetrics(
            tokenization_time=tokenize_time,
            model_forward_time=forward_time,
            pooling_time=pool_time,
            quantization_time=quantize_time,
            total_time=total_time,
            batch_size=len(texts),
            texts_per_second=len(texts)/total_time
        )
        
        return quantized_cpu.numpy(), sim, metrics

@dataclass
class ProcessingStats:
    """Track processing statistics."""
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
            
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            print(f"\nProgress update [{timestamp}]:")
            print(f"Recent TPS: {recent_tps:,.1f}")
            print(f"Overall TPS: {overall_tps:,.1f}")
            print(f"Processed: {self.total_processed:,} rows")
            print(f"Remaining: {self.total_remaining:,} rows")
            if self.total_remaining > 0 and overall_tps > 0:
                print(f"ETA: {self.get_eta_minutes():.1f} minutes")
            
            self.last_log_time = current_time
            self.last_processed = self.total_processed

def get_process_memory(pid):
    """Get memory usage for a process."""
    try:
        process = psutil.Process(pid)
        return process.memory_info().rss / (1024 * 1024)  # Convert to MB
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0

def get_gpu_memory():
    """Get MPS memory usage."""
    try:
        result = subprocess.run(['ps', '-o', 'rss=', '-p', str(os.getpid())], 
                              capture_output=True, text=True)
        return int(result.stdout.strip()) / 1024  # Convert KB to MB
    except:
        return 0

async def get_unprocessed_estimate(pool, last_id: int = 0) -> int:
    """Get current count of unprocessed rows."""
    async with pool.acquire() as conn:
        await conn.execute("SET statement_timeout = '60s'")
        if last_id == 0:
            result = await conn.fetchval("SELECT get_unprocessed_count()")
        else:
            result = await conn.fetchval("""
                SELECT count(*)
                FROM public.casts
                WHERE embedding384 IS NULL 
                AND id > $1
                AND text IS NOT NULL 
                AND length(trim(text)) > 0
            """, last_id)
        return int(result or 0)

class BatchProcessor:
    def __init__(self, pool, batch_size: int):
        self.pool = pool
        self.batch_size = batch_size
        self.prefetch_queue = asyncio.Queue(maxsize=PREFETCH_BATCHES)
        self.last_id = 0
        self.done = False
        self._prefetch_task: Optional[asyncio.Task] = None

    async def start(self):
        self._prefetch_task = asyncio.create_task(self.prefetch_worker())

    async def prefetch_worker(self):
        """Continuously prefetch next batches."""
        try:
            while not self.done:
                current_size = self.prefetch_queue.qsize()

                if current_size < PREFETCH_BATCHES:
                    batch, fetch_time = await fetch_next_batch(
                        self.pool,
                        self.last_id,
                        self.batch_size
                    )
                    if not batch:
                        # Check if completely done
                        remaining = await get_unprocessed_estimate(self.pool, self.last_id)
                        if remaining == 0:
                            break
                        continue

                    self.last_id = batch[-1]['id']
                    await self.prefetch_queue.put((batch, fetch_time))

                    if current_size == 0:
                        print(f"Prefetch: Queue refilled with {len(batch)} rows")
                await asyncio.sleep(0.01)
        except Exception as e:
            print(f"Error in prefetch worker: {str(e)}")
            self.done = True
            raise

    async def get_next_batch(self) -> Optional[Tuple[List[Dict[str, Any]], float]]:
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
        
        # Select query
        select_start = time.time()
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
        select_time = time.time() - select_start
        
        if not rows:
            return [], time.time() - fetch_start
            
        # Update timestamps
        update_start = time.time()
        ids = [row['id'] for row in rows]
        await conn.execute("""
            UPDATE casts
            SET embedding384_updated_at = NOW()
            WHERE id = ANY($1::bigint[])
        """, ids)
        update_time = time.time() - update_start
        
        return [dict(row) for row in rows], time.time() - fetch_start

class EmbeddingProcessor:
    """Handles model initialization and inference."""
    def __init__(self, batch_size: int, quiet: bool = True):
        self.model = OptimizedEmbeddingModel(batch_size, quiet=quiet)
        self.batch_size = batch_size
    
    def process_batch(self, texts: List[str], ids: List[int]) -> Tuple[np.ndarray, List[int], List[InferenceMetrics]]:
        """Process a batch of texts."""
        all_embeddings = []
        all_batch_ids = []
        all_metrics = []
        total_batches = len(texts) // self.batch_size + (1 if len(texts) % self.batch_size else 0)
        
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_ids = ids[i:i + self.batch_size]
            
            # Process batch and collect metrics
            embeddings, sim, metrics = self.model.encode(batch_texts)
            
            all_embeddings.append(embeddings)
            all_batch_ids.extend(batch_ids)
            all_metrics.append(metrics)
            
            # Log quartile statistics
            current_batch = i // self.batch_size + 1
            if current_batch % (total_batches // 4) == 0:
                print(f"\nProcessing {current_batch}/{total_batches} batches")
                print(f"Recent metrics:")
                print(f"  Tokenization: {metrics.tokenization_time:.3f}s")
                print(f"  Model Forward: {metrics.model_forward_time:.3f}s")
                print(f"  Pooling: {metrics.pooling_time:.3f}s")
                print(f"  Quantization: {metrics.quantization_time:.3f}s")
                print(f"  Total: {metrics.total_time:.3f}s")
                print(f"  Throughput: {metrics.texts_per_second:.1f} texts/sec")
            
            # Force garbage collection between batches
            gc.collect()
            torch.mps.empty_cache()
        
        # Combine all batches
        combined_embeddings = np.vstack(all_embeddings)
        return combined_embeddings, all_batch_ids, all_metrics

@dataclass
class BatchMetrics:
    """Track metrics for a single batch."""
    batch_size: int
    inference_time: float = 0
    quantize_time: float = 0
    db_select_time: float = 0
    db_update_time: float = 0
    total_time: float = 0
    memory_used: float = 0
    texts_per_second: float = 0

class PipelineLogger:
    """Consolidated logging for the embedding pipeline."""
    def __init__(self):
        self.start_time = time.time()
        self.last_log_time = self.start_time
        self.total_processed = 0
        self.total_remaining = 0
        self.batch_inference_metrics: List[InferenceMetrics] = []
        
    def set_total_remaining(self, total: int):
        self.total_remaining = total
        
    def log_event(self, message: str, force: bool = False):
        """Log a timestamped event."""
        now = time.time()
        if force or now - self.last_log_time >= 5:
            now_str = time.strftime("%Y-%m-%d %H:%M:%S")
            elapsed = now - self.start_time
            print(f"\n[{now_str} | +{elapsed:.1f}s] {message}")
            self.last_log_time = now
            
    def log_query_time(self, operation: str, duration: float):
        """Log database query timing."""
        self.log_event(f"{operation} query time: {duration:.3f}s", force=True)
        
    def log_batch_progress(self, current_batch: int, total_batches: int, metrics: InferenceMetrics):
        """Log progress after each sub-batch."""
        self.batch_inference_metrics.append(metrics)
        
        # Log every quarter batch
        if total_batches > 0 and current_batch % max(1, total_batches // 4) == 0:
            recent = self.batch_inference_metrics[-10:]  # Last 10 metrics
            
            self.log_event(
                f"Processing {current_batch}/{total_batches} batches\n"
                f"  Recent metrics:\n"
                f"    Tokenization: {np.mean([m.tokenization_time for m in recent]):.3f}s\n"
                f"    Model Forward: {np.mean([m.model_forward_time for m in recent]):.3f}s\n"
                f"    Pooling: {np.mean([m.pooling_time for m in recent]):.3f}s\n"
                f"    Quantization: {np.mean([m.quantization_time for m in recent]):.3f}s\n"
                f"    Total: {np.mean([m.total_time for m in recent]):.3f}s\n"
                f"    Throughput: {np.mean([m.texts_per_second for m in recent]):.1f} texts/sec",
                force=True
            )
            
    def update_processed(self, count: int):
        """Update progress counters and log if needed."""
        self.total_processed += count
        now = time.time()
        
        if now - self.last_log_time >= 5:
            elapsed = now - self.start_time
            rows_per_sec = self.total_processed / elapsed if elapsed > 0 else 0
            remaining = max(0, self.total_remaining - self.total_processed)
            
            status = (
                f"Processed: {self.total_processed:,} / {self.total_remaining:,} rows\n"
                f"Elapsed: {elapsed/60:.1f} minutes, Overall TPS: {rows_per_sec:,.1f}"
            )
            
            if rows_per_sec > 0:
                eta_min = (remaining / rows_per_sec) / 60
                status += f"\nETA: {eta_min:.1f} minutes for {remaining:,} rows"
                
            self.log_event(status)

async def update_embeddings(pool, batch_ids: List[int], embeddings: np.ndarray) -> Tuple[float, int]:
    """Async function to update embeddings in the database. Returns (time_taken, rows_updated)"""
    start_time = time.time()
    chunk_size = CHUNK_SIZE
    total_updated = 0
    
    for i in range(0, len(batch_ids), chunk_size):
        chunk_ids = batch_ids[i:i + chunk_size]
        chunk_embeddings = embeddings[i:i + chunk_size]
        
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
                    RETURNING t.id
                """
                result = await conn.fetch(update_sql)
                rows_updated = len(result)
                total_updated += rows_updated
                
                if rows_updated != len(chunk_ids):
                    print(f"\nWarning: Expected to update {len(chunk_ids)} rows but updated {rows_updated}")
    
    return time.time() - start_time, total_updated

@dataclass
class BatchSummary:
    """Track total time spent in each operation for a batch."""
    batch_size: int = 0
    fetch_time: float = 0
    embedding_time: float = 0
    db_update_time: float = 0
    total_time: float = 0
    start_time: float = field(default_factory=time.time)
    update_complete: asyncio.Event = field(default_factory=asyncio.Event)
    
    def log_summary(self):
        """Log summary of all operations for this batch."""
        total_time = time.time() - self.start_time
        print(f"\nBatch Summary ({self.batch_size:,} rows):")
        print(f"  Fetch Time:     {self.fetch_time:.1f}s ({(self.fetch_time/total_time)*100:.1f}%)")
        print(f"  Embedding Time: {self.embedding_time:.1f}s ({(self.embedding_time/total_time)*100:.1f}%)")
        print(f"  DB Update Time: {self.db_update_time:.1f}s ({(self.db_update_time/total_time)*100:.1f}%)")
        print(f"  Total Time:     {total_time:.1f}s")
        print(f"  Throughput:     {self.batch_size/total_time:.1f} rows/sec")

async def db_update_worker(pool, queue: asyncio.Queue, stats: ProcessingStats):
    """Worker to handle async DB updates."""
    last_log = time.time()
    updates_since_log = 0
    total_update_time = 0
    
    while True:
        item = await queue.get()
        if item is None:  # Poison pill
            break
            
        batch_ids, embeddings, batch_summary = item
        start_time = time.time()
        update_time, rows_updated = await update_embeddings(pool, batch_ids, embeddings)
        
        # Update batch summary with actual DB time
        batch_summary.db_update_time = update_time
        batch_summary.update_complete.set()
        
        total_update_time += update_time
        updates_since_log += rows_updated
        
        # Log DB update stats every 5 seconds
        now = time.time()
        if now - last_log >= 5:
            avg_time = total_update_time / (updates_since_log or 1)
            print(f"\nDB Update Stats:")
            print(f"  Queue Size: {queue.qsize():,} batches")
            print(f"  Recent Updates: {updates_since_log:,} rows")
            print(f"  Avg Update Time: {avg_time:.3f}s per batch")
            print(f"  Total Update Time: {total_update_time:.1f}s")
            
            # Reset counters
            last_log = now
            updates_since_log = 0
            total_update_time = 0
        
        stats.update(rows_updated)
        queue.task_done()

@dataclass
class BatchStats:
    """Track statistics for an entire batch."""
    inference_metrics: List[InferenceMetrics] = field(default_factory=list)
    db_select_time: float = 0
    db_update_time: float = 0
    total_time: float = 0
    start_time: float = field(default_factory=time.time)
    last_log_time: float = field(default_factory=time.time)
    total_processed: int = 0
    
    def log_quartile(self, current_batch: int, total_batches: int, processed: int, total: int):
        """Log statistics at quartile boundaries."""
        if current_batch % (total_batches // 4) == 0:
            metrics = self.inference_metrics[-10:]  # Last 10 sub-batches
            current_time = time.strftime('%Y-%m-%d %H:%M:%S')
            elapsed = time.time() - self.start_time
            
            # Calculate key metrics
            progress = (processed / total) * 100 if total > 0 else 0
            rows_per_sec = processed / elapsed if elapsed > 0 else 0
            remaining_rows = total - processed
            eta_minutes = (remaining_rows / rows_per_sec) / 60 if rows_per_sec > 0 else 0
            
            print(f"\n{'='*80}")
            print(f"Quarter-Batch Analysis ({current_batch}/{total_batches}) - {current_time}")
            print(f"{'='*80}")
            
            print("\nProgress Summary:")
            print(f"  Processed:      {processed:,}/{total:,} rows ({progress:.1f}%)")
            print(f"  Elapsed Time:   {elapsed/60:.1f} minutes")
            print(f"  Overall Speed:  {rows_per_sec:.1f} rows/sec")
            print(f"  ETA:           {eta_minutes:.1f} minutes")
            
            print("\nRecent Performance (last 10 batches):")
            print(f"  Tokenization:   {np.mean([m.tokenization_time for m in metrics]):.3f}s / {np.percentile([m.tokenization_time for m in metrics], 95):.3f}s")
            print(f"  Model Forward:  {np.mean([m.model_forward_time for m in metrics]):.3f}s / {np.percentile([m.model_forward_time for m in metrics], 95):.3f}s")
            print(f"  Pooling:       {np.mean([m.pooling_time for m in metrics]):.3f}s / {np.percentile([m.pooling_time for m in metrics], 95):.3f}s")
            print(f"  Quantization:  {np.mean([m.quantization_time for m in metrics]):.3f}s / {np.percentile([m.quantization_time for m in metrics], 95):.3f}s")
            print(f"  Total:         {np.mean([m.total_time for m in metrics]):.3f}s / {np.percentile([m.total_time for m in metrics], 95):.3f}s")
            
            print("\nThroughput Analysis:")
            print(f"  Recent TPS:    {np.mean([m.texts_per_second for m in metrics]):.1f} texts/sec")
            print(f"  Peak TPS:      {np.max([m.texts_per_second for m in metrics]):.1f} texts/sec")
            print(f"  Current TPS:   {rows_per_sec:.1f} rows/sec")
            
            print("\nDatabase Performance:")
            print(f"  Select Time:   {self.db_select_time:.3f}s")
            print(f"  Update Time:   {self.db_update_time:.3f}s")
            
            print("\nMemory Usage:")
            print(f"  Process:       {get_process_memory(os.getpid()):.1f}MB")
            print(f"  GPU:           {get_gpu_memory():.1f}MB")
            print(f"{'='*80}\n")
            
            self.last_log_time = time.time()
            
    def update_processed(self, count: int):
        """Update the total processed count."""
        self.total_processed += count

async def process_casts(pool, batch_size: int):
    stats = ProcessingStats()
    stats.total_remaining = await get_unprocessed_estimate(pool)
    print(f"Starting processing of {stats.total_remaining:,} rows...")
    
    # Initialize processor and queues
    processor = BatchProcessor(pool, BATCH_SIZE_ROWS)
    MAX_QUEUE_SIZE = 10  # Maximum number of unprocessed batches to allow
    db_queue = asyncio.Queue()
    await processor.start()
    
    # Start DB update worker
    update_worker = asyncio.create_task(db_update_worker(pool, db_queue, stats))
    
    embedding_processor = EmbeddingProcessor(batch_size, quiet=True)
    batch_stats = BatchStats()
    processed_rows = 0
    
    try:
        while True:
            batch_summary = BatchSummary()
            
            # Check queue size and wait if too large
            while db_queue.qsize() >= MAX_QUEUE_SIZE:
                print(f"\nWaiting for DB queue to drain. Current size: {db_queue.qsize()}")
                await asyncio.sleep(1)
            
            # Get batch with timing
            fetch_start = time.time()
            batch, fetch_time = await processor.get_next_batch()
            batch_summary.fetch_time = time.time() - fetch_start
            
            if not batch:
                remaining = await get_unprocessed_estimate(pool, processor.last_id)
                if remaining == 0:
                    break
                continue
            
            batch_summary.batch_size = len(batch)
            texts = [cast['text'] for cast in batch]
            ids = [cast['id'] for cast in batch]
            
            # Process embeddings
            embed_start = time.time()
            embeddings, batch_ids, metrics = embedding_processor.process_batch(texts, ids)
            batch_summary.embedding_time = time.time() - embed_start
            batch_stats.inference_metrics.extend(metrics)
            
            # Queue DB update with batch summary
            await db_queue.put((batch_ids, embeddings, batch_summary))
            
            processed_rows += len(batch_ids)
            batch_stats.update_processed(len(batch_ids))
            
            # Wait for DB update to complete before logging summary
            await batch_summary.update_complete.wait()
            batch_summary.log_summary()
            
            # Log quarter batch analysis
            current_batch = processed_rows // batch_size
            total_batches = stats.total_remaining // batch_size
            batch_stats.log_quartile(current_batch, total_batches, processed_rows, stats.total_remaining)
    
    finally:
        processor.done = True
        if processor._prefetch_task:
            await processor._prefetch_task
        
        # Wait for remaining DB updates and stop worker
        await db_queue.put(None)
        await update_worker
        
        # Final stats
        total_time = time.time() - batch_stats.start_time
        print("\nProcessing complete!")
        print(f"Total processed: {batch_stats.total_processed:,} rows")
        print(f"Overall TPS: {batch_stats.total_processed/total_time:,.1f}")
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
        
        await process_casts(db.pool, BATCH_SIZE)
        
        total_duration = time.time() - start_time
        print(f"\nTotal processing time: {total_duration:.1f}s")
        
    finally:
        await db.close_pool()

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    asyncio.run(main()) 