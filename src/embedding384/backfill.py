"""
Batch process embedding generation for casts table on Apple Silicon (MPS).
Uses async prefetching and parallel processing with PyTorch optimizations.
Leverages MPS acceleration, parallel tokenization, and float16 precision.
Stores embeddings as int8 vectors for efficiency.
"""

import os
import sys
import time
import gc
import psutil
import tracemalloc
import asyncio
import warnings
import subprocess
import multiprocessing
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass, field
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
import logging

from src.db.connect import db

# --------------------------------------------------------------------------------------
# Basic Configuration
# --------------------------------------------------------------------------------------
BATCH_SIZE = 4112                # Embedding batch size
BATCH_SIZE_ROWS = 200000         # Number of rows to process per instance
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_SEQ_LENGTH = 256             # Maximum sequence length for tokenization
CHUNK_SIZE = 25000               # Rows to update in one transaction chunk
DB_TIMEOUT = 400                 # 5-minute timeout for database operations

# Set fast math for MPS (allows faster approximations)
os.environ["PYTORCH_MPS_FAST_MATH"] = "1"

load_dotenv()  # Load any local .env variables
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", message=".*_is_quantized_training_enabled.*")
warnings.filterwarnings("ignore", message=r".*loss_type=None.*Unrecognised.*", category=UserWarning)

# Structured logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------
# Data Classes and Helpers
# --------------------------------------------------------------------------------------
@dataclass
class InferenceMetrics:
    tokenization_time: float
    model_forward_time: float
    pooling_time: float
    quantization_time: float
    total_time: float
    batch_size: int
    texts_per_second: float
    cosine_sim: float

@dataclass
class ProcessingStats:
    start_time: float = field(default_factory=time.time)
    total_processed: int = 0
    total_remaining: int = 0
    last_log_time: float = field(default_factory=time.time)
    last_processed: int = 0
    
    def get_recent_tps(self) -> float:
        current_time = time.time()
        time_since_last = current_time - self.last_log_time
        if time_since_last > 0:
            return (self.total_processed - self.last_processed) / time_since_last
        return 0.0
    
    def get_eta_minutes(self) -> float:
        current_time = time.time()
        total_time = current_time - self.start_time
        if self.total_processed and total_time:
            overall_tps = self.total_processed / total_time
            if overall_tps > 0:
                return (self.total_remaining / overall_tps) / 60
        return 0.0
    
    def update(self, processed: int):
        self.total_processed += processed
        self.total_remaining = max(0, self.total_remaining - processed)
        
        current_time = time.time()
        time_since_last = current_time - self.last_log_time
        if time_since_last >= 5:
            total_time = current_time - self.start_time
            recent_tps = self.get_recent_tps()
            overall_tps = (self.total_processed / total_time) if total_time else 0
            
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            logger.info(f"Progress update [{timestamp}]:")
            logger.info(f"  Recent TPS: {recent_tps:,.1f}")
            logger.info(f"  Overall TPS: {overall_tps:,.1f}")
            logger.info(f"  Processed: {self.total_processed:,} rows")
            logger.info(f"  Remaining: {self.total_remaining:,} rows")
            if self.total_remaining > 0 and overall_tps > 0:
                logger.info(f"  ETA: {self.get_eta_minutes():.1f} minutes")
            
            self.last_log_time = current_time
            self.last_processed = self.total_processed


@dataclass
class BatchSummary:
    batch_size: int = 0
    fetch_time: float = 0.0
    embedding_time: float = 0.0
    tokenization_time: float = 0.0
    forward_time: float = 0.0
    pooling_time: float = 0.0
    quantization_time: float = 0.0
    db_update_time: float = 0.0
    avg_cosine_sim: float = 0.0
    start_time: float = field(default_factory=time.time)
    
    def log_summary(self):
        total_time = self.fetch_time + self.embedding_time + self.db_update_time
        logger.info(f"Batch Summary ({self.batch_size:,} rows):")
        if total_time <= 0:
            return
        logger.info(f"  Fetch Time:     {self.fetch_time:.1f}s ({(self.fetch_time/total_time)*100:.1f}%)")
        logger.info(f"  Embedding Time: {self.embedding_time:.1f}s ({(self.embedding_time/total_time)*100:.1f}%)")
        if self.embedding_time > 0:
            logger.info(f"    → Tokenization:  {self.tokenization_time:.1f}s ({(self.tokenization_time/self.embedding_time)*100:.1f}%)")
            logger.info(f"    → Model Forward: {self.forward_time:.1f}s ({(self.forward_time/self.embedding_time)*100:.1f}%)")
            logger.info(f"    → Pooling:      {self.pooling_time:.1f}s ({(self.pooling_time/self.embedding_time)*100:.1f}%)")
            logger.info(f"    → Quantization: {self.quantization_time:.1f}s ({(self.quantization_time/self.embedding_time)*100:.1f}%)")
        logger.info(f"  DB Update Time: {self.db_update_time:.1f}s ({(self.db_update_time/total_time)*100:.1f}%)")
        logger.info(f"  Total Time:     {total_time:.1f}s")
        logger.info(f"  Throughput:     {self.batch_size/total_time:.1f} rows/sec")
        logger.info(f"  Cosine Sim:     {self.avg_cosine_sim:.4f}")


class OperationTiming:
    def __init__(self):
        self.operation_start = time.time()
        self.fetch_start = 0.0
        self.fetch_end = 0.0
        self.inference_start = 0.0
        self.inference_end = 0.0
        self.db_update_start = 0.0
        self.db_update_end = 0.0

    def log_overlap(self):
        logger.info("Operation Timing:")
        fetch_time = self.fetch_end - self.fetch_start
        inference_time = self.inference_end - self.inference_start
        update_time = self.db_update_end - self.db_update_start
        fetch_inference_gap = self.inference_start - self.fetch_end
        inference_update_gap = self.db_update_start - self.inference_end
        
        logger.info(f"  Fetch:     {self.fetch_start:.1f}s → {self.fetch_end:.1f}s ({fetch_time:.1f}s)")
        logger.info(f"  Inference: {self.inference_start:.1f}s → {self.inference_end:.1f}s ({inference_time:.1f}s)")
        logger.info(f"  DB Update: {self.db_update_start:.1f}s → {self.db_update_end:.1f}s ({update_time:.1f}s)")
        logger.info("  Gaps:")
        logger.info(f"    Fetch→Inference: {fetch_inference_gap:.3f}s")
        logger.info(f"    Inference→Update: {inference_update_gap:.3f}s")


@dataclass
class BatchStats:
    inference_metrics: List[InferenceMetrics] = field(default_factory=list)
    db_select_time: float = 0
    db_update_time: float = 0
    total_time: float = 0
    start_time: float = field(default_factory=time.time)
    last_log_time: float = field(default_factory=time.time)
    total_processed: int = 0
    
    def log_quartile(self, current_batch: int, total_batches: int, processed: int, total: int):
        if total_batches <= 0:
            return
            
        # Log every quarter of total batches
        if current_batch % max(1, total_batches // 4) == 0:
            metrics = self.inference_metrics[-10:]
            current_time = time.strftime('%Y-%m-%d %H:%M:%S')
            elapsed = time.time() - self.start_time
            progress = (processed / total) * 100 if total else 0
            rows_per_sec = processed / elapsed if elapsed else 0
            remaining_rows = total - processed
            eta_minutes = (remaining_rows / rows_per_sec) / 60 if rows_per_sec > 0 else 0
            
            logger.info("="*80)
            logger.info(f"Quarter-Batch Analysis ({current_batch}/{total_batches}) - {current_time}")
            logger.info("="*80)
            
            logger.info("Progress Summary:")
            logger.info(f"  Processed:      {processed:,}/{total:,} rows ({progress:.1f}%)")
            logger.info(f"  Elapsed Time:   {elapsed/60:.1f} minutes")
            logger.info(f"  Overall Speed:  {rows_per_sec:.1f} rows/sec")
            logger.info(f"  ETA:            {eta_minutes:.1f} minutes")
            
            if metrics:
                logger.info("\nPerformance Breakdown (last 10 batches):")
                t_times = [m.tokenization_time for m in metrics]
                f_times = [m.model_forward_time for m in metrics]
                p_times = [m.pooling_time for m in metrics]
                q_times = [m.quantization_time for m in metrics]
                tot_times = [m.total_time for m in metrics]
                
                logger.info(
                    f"  Tokenization:   {np.mean(t_times):.3f}s / "
                    f"{np.percentile(t_times, 95):.3f}s (avg/p95)"
                )
                logger.info(
                    f"  Model Forward:  {np.mean(f_times):.3f}s / "
                    f"{np.percentile(f_times, 95):.3f}s (avg/p95)"
                )
                logger.info(
                    f"  Pooling:        {np.mean(p_times):.3f}s / "
                    f"{np.percentile(p_times, 95):.3f}s (avg/p95)"
                )
                logger.info(
                    f"  Quantization:   {np.mean(q_times):.3f}s / "
                    f"{np.percentile(q_times, 95):.3f}s (avg/p95)"
                )
                logger.info(
                    f"  Total:          {np.mean(tot_times):.3f}s / "
                    f"{np.percentile(tot_times, 95):.3f}s (avg/p95)"
                )
                
                logger.info("\nOptimization Impact:")
                all_cosine_sim = [m.cosine_sim for m in self.inference_metrics if m.cosine_sim > 0]
                if all_cosine_sim:
                    logger.info(
                        f"  Cosine Similarity: {np.mean(all_cosine_sim):.4f} (avg), "
                        f"{np.min(all_cosine_sim):.4f} (min), {np.max(all_cosine_sim):.4f} (max)"
                    )
                
                total_process_time = (
                    np.sum(t_times) + np.sum(f_times) + np.sum(p_times) + np.sum(q_times)
                )
                if total_process_time > 0:
                    tokenize_pct = np.sum(t_times) / total_process_time * 100
                    forward_pct = np.sum(f_times) / total_process_time * 100
                    pooling_pct = np.sum(p_times) / total_process_time * 100
                    quantize_pct = np.sum(q_times) / total_process_time * 100
                    logger.info("\nTime Distribution:")
                    logger.info(f"  Tokenization:   {tokenize_pct:.1f}%")
                    logger.info(f"  Model Forward:  {forward_pct:.1f}%")
                    logger.info(f"  Pooling:        {pooling_pct:.1f}%")
                    logger.info(f"  Quantization:   {quantize_pct:.1f}%")
                
                logger.info("\nThroughput Analysis:")
                logger.info(f"  Recent TPS:    {np.mean([m.texts_per_second for m in metrics]):.1f} texts/sec")
                logger.info(f"  Peak TPS:      {np.max([m.texts_per_second for m in metrics]):.1f} texts/sec")
                logger.info(f"  Current TPS:   {rows_per_sec:.1f} rows/sec")
            
            logger.info("\nDatabase Performance:")
            logger.info(f"  Select Time:   {self.db_select_time:.3f}s")
            logger.info(f"  Update Time:   {self.db_update_time:.3f}s")
            
            logger.info("\nMemory Usage:")
            logger.info(f"  Process:       {get_process_memory(os.getpid()):.1f}MB")
            logger.info("="*80)
            
            self.last_log_time = time.time()
    
    def update_processed(self, count: int):
        self.total_processed += count


# --------------------------------------------------------------------------------------
# Memory Utilities (Process memory only, since we're on MPS)
# --------------------------------------------------------------------------------------
def get_process_memory(pid):
    """Return the resident set size in MB."""
    try:
        process = psutil.Process(pid)
        return process.memory_info().rss / (1024 * 1024)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0

def detailed_memory_usage(tracemalloc_enabled=False):
    """
    Returns a dict with process memory usage stats,
    plus top allocations if tracemalloc is enabled.
    """
    process = psutil.Process(os.getpid())
    process_info = process.memory_info()
    
    # MPS usage not exposed directly like CUDA, so we just show RSS + tracemalloc
    mps_memory = 0
    
    tracemalloc_info = None
    if tracemalloc_enabled and tracemalloc.is_tracing():
        snapshot = tracemalloc.take_snapshot()
        tracemalloc_info = snapshot.statistics('lineno')[:5]
    
    return {
        'rss': process_info.rss / 1024**2,
        'vms': process_info.vms / 1024**2,
        'shared': getattr(process_info, 'shared', 0) / 1024**2,
        'mps_memory': mps_memory,
        'tracemalloc': tracemalloc_info,
    }


# --------------------------------------------------------------------------------------
# Model for MPS
# --------------------------------------------------------------------------------------
class OptimizedEmbeddingModel:
    """
    A minimal model class designed for Apple Silicon with MPS.
    """
    def __init__(self, batch_size: int, quiet: bool = False):
        self.device = torch.device("mps")
        self.batch_size = batch_size
        self.quiet = quiet
        self.model_name = MODEL_NAME
        
        if not quiet:
            logger.info(f"Initializing model with batch_size={batch_size} on MPS...")
            logger.info("Using PyTorch native optimizations")
        
        self._initialize_model()
        
    def _initialize_model(self):
        logger.info("Initializing model...")
        init_start = time.time()
        torch.set_num_threads(multiprocessing.cpu_count())
        gc.collect()
        
        # Tokenizer with parallel CPU usage
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer_executor = ThreadPoolExecutor(max_workers=os.cpu_count())
        
        # Load the base model in float16 on MPS
        logger.info("Loading model (float16) for MPS...")
        base_model = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        base_model.to(self.device)
        base_model.eval()
        
        self.model = base_model
        
        # Over-allocate input buffers by ~10%
        buffer_size = int(self.batch_size * 1.1)
        self.input_buffers = {
            'input_ids': torch.zeros((buffer_size, MAX_SEQ_LENGTH), dtype=torch.long, device=self.device),
            'attention_mask': torch.zeros((buffer_size, MAX_SEQ_LENGTH), dtype=torch.long, device=self.device)
        }
        
        # Output buffer
        self.output_buffer = torch.zeros((self.batch_size, 384), dtype=torch.float16, device=self.device)
        
        # Warmup
        sample_texts = ["This is a sample text for warmup"] * min(32, self.batch_size)
        _ = self.encode(sample_texts, quantize=True)
        torch.mps.empty_cache()
        
        logger.info(f"✨ Model initialization completed in {time.time() - init_start:.1f}s")
    
    def _mean_pooling(self, model_output, attention_mask):
        # More concise PyTorch pooling
        token_embeddings = model_output['last_hidden_state']
        mask_expanded = attention_mask.unsqueeze(-1).float()
        masked_embeddings = token_embeddings * mask_expanded
        pooled = masked_embeddings.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
        return torch.nn.functional.normalize(pooled, p=2, dim=1)
    
    def encode(
        self,
        texts: List[str],
        quantize: bool = True
    ) -> Tuple[Union[torch.Tensor, np.ndarray], float, InferenceMetrics]:
        start_time = time.time()
        tokenize_start = time.time()
        
        # Function for parallel tokenization
        def tokenize_subset(text_subset: List[str]):
            return self.tokenizer(
                text_subset,
                padding='max_length',
                truncation=True,
                max_length=MAX_SEQ_LENGTH,
                return_tensors="pt"
            )
        
        # Parallel tokenization if large batch
        if len(texts) > 32:
            chunk_size = max(8, len(texts) // min(os.cpu_count(), 8))
            text_chunks = [texts[i:i+chunk_size] for i in range(0, len(texts), chunk_size)]
            tokenized_chunks = list(self.tokenizer_executor.map(tokenize_subset, text_chunks))
            inputs = {
                'input_ids': torch.cat([c['input_ids'] for c in tokenized_chunks]),
                'attention_mask': torch.cat([c['attention_mask'] for c in tokenized_chunks])
            }
        else:
            inputs = tokenize_subset(texts)
        
        tokenize_time = time.time() - tokenize_start
        
        # Copy to pre-allocated buffers
        for k, v in inputs.items():
            if k in self.input_buffers:
                if (v.size(0) > self.input_buffers[k].size(0)
                        or v.size(1) > self.input_buffers[k].size(1)):
                    # Resize if needed
                    self.input_buffers[k] = torch.zeros(
                        (max(v.size(0), self.input_buffers[k].size(0)),
                         max(v.size(1), self.input_buffers[k].size(1))),
                        dtype=self.input_buffers[k].dtype,
                        device=self.device
                    )
                self.input_buffers[k][:v.size(0), :v.size(1)] = v.to(self.device)
                inputs[k] = self.input_buffers[k][:v.size(0), :v.size(1)]
        
        with torch.inference_mode():
            forward_start = time.time()
            outputs = self.model(inputs['input_ids'], inputs['attention_mask'])
            torch.mps.synchronize()
            forward_time = time.time() - forward_start
            
            pool_start = time.time()
            embeddings = self._mean_pooling(outputs, inputs['attention_mask'])
            # Resize output buffer if necessary
            if embeddings.size(0) > self.output_buffer.size(0):
                self.output_buffer = torch.zeros((embeddings.size(0), embeddings.shape[1]),
                                                 dtype=torch.float16,
                                                 device=self.device)
            self.output_buffer[:embeddings.size(0)] = embeddings
            float16_embeddings = self.output_buffer[:embeddings.size(0)]
            torch.mps.synchronize()
            pool_time = time.time() - pool_start
            
            # If no quantization, return float16
            if not quantize:
                total_time = time.time() - start_time
                return (
                    float16_embeddings,
                    0.0,
                    InferenceMetrics(
                        tokenization_time=tokenize_time,
                        model_forward_time=forward_time,
                        pooling_time=pool_time,
                        quantization_time=0.0,
                        total_time=total_time,
                        batch_size=len(texts),
                        texts_per_second=len(texts)/total_time if total_time else 0.0,
                        cosine_sim=0.0
                    )
                )
            
            # Quantize
            quantize_start = time.time()
            global_max_abs = float16_embeddings.abs().max()
            global_scale = global_max_abs / 127.0
            scaled_embeddings = float16_embeddings / global_scale
            rounded_embeddings = scaled_embeddings.round().clamp(-128, 127)
            quantized_mps = rounded_embeddings.to(torch.int8)
            quantized_cpu = quantized_mps.detach().cpu()
            
            # Cosine similarity check
            dequantized = quantized_mps.float() * global_scale
            sim = float(
                torch.nn.functional.cosine_similarity(
                    float16_embeddings, dequantized, dim=1
                ).mean().item()
            )
            torch.mps.synchronize()
            quantize_time = time.time() - quantize_start
        
        total_time = time.time() - start_time
        metrics = InferenceMetrics(
            tokenization_time=tokenize_time,
            model_forward_time=forward_time,
            pooling_time=pool_time,
            quantization_time=quantize_time,
            total_time=total_time,
            batch_size=len(texts),
            texts_per_second=len(texts)/total_time if total_time else 0.0,
            cosine_sim=sim
        )
        
        return quantized_cpu.numpy(), sim, metrics


# --------------------------------------------------------------------------------------
# DB Processing
# --------------------------------------------------------------------------------------
async def get_unprocessed_estimate(pool) -> int:
    start_time = time.time()
    async with pool.acquire() as conn:
        # rely on the asyncpg pool's command_timeout=60
        result = await conn.fetchval("""
            SELECT COUNT(*) 
                FROM farcaster.casts 
                WHERE embedding384 IS NULL 
                AND text IS NOT NULL
                AND text != ''
        """, timeout=DB_TIMEOUT)
        query_time = time.time() - start_time
        logger.info(f"Unprocessed count query took {query_time:.2f}s, found {int(result or 0):,} rows")
        return int(result or 0)


async def reset_stale_rows(pool) -> int:
    """
    Optionally resets rows that never completed embedding due to a prior crash.
    """
    async with pool.acquire() as conn:
        async with conn.transaction():
            result = await conn.fetch("""
                UPDATE farcaster.casts 
                SET embedding384_updated_at = NULL
                WHERE embedding384 IS NULL 
                    AND embedding384_updated_at IS NOT NULL 
                    AND embedding384_updated_at < NOW() - interval '5 minutes'
                    AND text IS NOT NULL
                    AND text != ''
                RETURNING id
            """, timeout=DB_TIMEOUT)
            return len(result)


async def fetch_next_batch(pool, limit: int = 200000, retries: int = 3) -> Tuple[List[Dict[str, Any]], float]:
    """
    Fetch the next batch with retry logic and exponential backoff.
    Uses direct ordering by ID for simplicity and efficiency.
    """
    for attempt in range(retries):
        try:
            fetch_start = time.time()
            async with pool.acquire() as conn:
                async with conn.transaction():
                    rows = await conn.fetch("""
                        WITH selected AS (
                            SELECT id, text
                            FROM farcaster.casts
                            WHERE embedding384 IS NULL 
                                AND text IS NOT NULL 
                                AND text != ''
                                AND (embedding384_updated_at IS NULL OR embedding384_updated_at < NOW() - interval '1 hour')
                            -- No randomization needed - use your index directly
                            LIMIT $1
                            FOR UPDATE SKIP LOCKED
                        )
                        UPDATE farcaster.casts c
                        SET embedding384_updated_at = NOW()
                        FROM selected s
                        WHERE c.id = s.id
                        RETURNING c.id, c.text
                    """, limit, timeout=DB_TIMEOUT)
                    
            total_time = time.time() - fetch_start
            if not rows:
                return [], total_time
            
            logger.info("Fetch Timing:")
            logger.info(f"  Total Time: {total_time:.3f}s")
            logger.info(f"  Rows: {len(rows):,}")
            return [dict(r) for r in rows], total_time
        
        except Exception as e:
            logger.warning(f"Fetch failed (attempt {attempt+1}/{retries}): {e}")
            if attempt == retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)
    
    return [], 0.0


class BatchProcessor:
    """
    Orchestrates fetching next batch of rows for embedding.
    """
    def __init__(self, pool, batch_size: int):
        self.pool = pool
        self.batch_size = batch_size
        self.done = False

    async def start(self):
        pass  # no-op

    async def get_next_batch(self) -> Optional[Tuple[List[Dict[str, Any]], float]]:
        if self.done:
            return None
        batch, fetch_time = await fetch_next_batch(self.pool, self.batch_size)
        if not batch:
            remaining = await get_unprocessed_estimate(self.pool)
            if remaining == 0:
                self.done = True
            return None
        return batch, fetch_time


class EmbeddingProcessor:
    """
    Wraps the model to process a batch of texts, returning quantized embeddings.
    """
    def __init__(self, batch_size: int, quiet: bool = True):
        self.model = OptimizedEmbeddingModel(batch_size, quiet=quiet)
        self.batch_size = batch_size
    
    def process_batch(
        self,
        texts: List[str],
        ids: List[int]
    ) -> Tuple[np.ndarray, List[int], List[InferenceMetrics]]:
        all_float16_embeddings = []
        all_batch_ids = []
        all_metrics = []
        
        if len(texts) <= self.batch_size * 2:
            optimal_batch_size = len(texts)
            total_batches = 1
        else:
            optimal_batch_size = self.batch_size
            total_batches = len(texts) // optimal_batch_size + (1 if len(texts) % optimal_batch_size else 0)
        
        progress_updates = set([int(total_batches * i / 20) for i in range(1, 20)] + [total_batches])
        last_progress_print = 0
        
        bar_length = 30
        logger.info(f"Processing: [{'-'*bar_length}] 0.0% (0/{total_batches} batches) | Preparing...")
        
        # Go through batches in texts
        for i in range(0, len(texts), optimal_batch_size):
            batch_texts = texts[i:i + optimal_batch_size]
            batch_ids = ids[i:i + optimal_batch_size]
            
            float16_emb = None
            metrics = None
            
            # Retry logic in case of transient errors
            for attempt in range(3):
                try:
                    float16_emb, _, metrics = self.model.encode(batch_texts, quantize=False)
                    break
                except Exception as e:
                    logger.warning(f"Batch processing failed (attempt {attempt+1}/3): {e}")
                    if attempt == 2:
                        raise
                    time.sleep(2 ** attempt)
            
            if float16_emb is None or not metrics:
                continue
            
            all_float16_embeddings.append(float16_emb)
            all_batch_ids.extend(batch_ids)
            all_metrics.append(metrics)
            
            current_batch = i // optimal_batch_size + 1
            if current_batch in progress_updates or current_batch - last_progress_print >= 10:
                progress = current_batch / total_batches
                filled_length = int(bar_length * progress)
                bar = '=' * filled_length + '-' * (bar_length - filled_length)
                logger.info(
                    f"Processing: [{bar}] {progress*100:.1f}% "
                    f"({current_batch}/{total_batches} batches) | "
                    f"Speed: {metrics.texts_per_second:.0f} texts/s | "
                    f"Batch: {len(batch_texts)} texts"
                )
                last_progress_print = current_batch
        
        logger.info("Combining embeddings and quantizing...")
        total_embeddings = sum(emb.size(0) for emb in all_float16_embeddings)
        
        # If huge total, do chunk quantization
        if total_embeddings > 100000 and len(all_float16_embeddings) > 1:
            logger.info(f"Large batch detected ({total_embeddings} embeddings). Processing in chunks...")
            embedding_dim = all_float16_embeddings[0].size(1)
            combined_quantized = np.zeros((total_embeddings, embedding_dim), dtype=np.int8)
            start_idx = 0
            quantize_start = time.time()
            
            for float16_emb in all_float16_embeddings:
                chunk_size = float16_emb.size(0)
                emb_cpu = float16_emb.detach().cpu().float()
                global_max_abs = emb_cpu.abs().max()
                global_scale = global_max_abs / 127.0
                quantized_cpu = (emb_cpu / global_scale).round().clamp(-128, 127).to(torch.int8).numpy()
                combined_quantized[start_idx:start_idx+chunk_size] = quantized_cpu
                start_idx += chunk_size
                del float16_emb
                torch.mps.empty_cache()
            
            all_float16_embeddings.clear()
            torch.mps.empty_cache()
            
            sim = 0.998  # approximate for large chunk
            quantize_time = time.time() - quantize_start
            
        else:
            # Single-step quantization
            combined_float16 = torch.cat(all_float16_embeddings, dim=0)
            quantize_start = time.time()
            global_max_abs = combined_float16.abs().max()
            global_scale = global_max_abs / 127.0
            scaled = combined_float16 / global_scale
            rounded = scaled.round().clamp(-128, 127)
            quantized_mps = rounded.to(torch.int8)
            dequantized = quantized_mps.float() * global_scale
            sim = float(torch.nn.functional.cosine_similarity(combined_float16, dequantized, dim=1).mean().item())
            combined_quantized = quantized_mps.detach().cpu().numpy()
            quantize_time = time.time() - quantize_start
        
        # Update metrics with quantization times
        quantize_per_batch = quantize_time / len(all_metrics) if all_metrics else 0
        for metric in all_metrics:
            metric.quantization_time = quantize_per_batch
            metric.total_time += quantize_per_batch
            metric.texts_per_second = metric.batch_size / metric.total_time if metric.total_time else 0
            metric.cosine_sim = sim
        
        if total_batches > 0:
            avg_metrics = InferenceMetrics(
                tokenization_time=np.mean([m.tokenization_time for m in all_metrics]),
                model_forward_time=np.mean([m.model_forward_time for m in all_metrics]),
                pooling_time=np.mean([m.pooling_time for m in all_metrics]),
                quantization_time=np.mean([m.quantization_time for m in all_metrics]),
                total_time=np.mean([m.total_time for m in all_metrics]),
                batch_size=np.mean([m.batch_size for m in all_metrics]),
                texts_per_second=np.mean([m.texts_per_second for m in all_metrics]),
                cosine_sim=np.mean([m.cosine_sim for m in all_metrics])
            )
            logger.info("Batch Processing Summary:")
            logger.info(f"  Average Speed: {avg_metrics.texts_per_second:.0f} texts/s")
            logger.info(
                f"  Times (avg): tokenize={avg_metrics.tokenization_time:.3f}s, "
                f"forward={avg_metrics.model_forward_time:.3f}s, "
                f"pool={avg_metrics.pooling_time:.3f}s, "
                f"quantize={avg_metrics.quantization_time:.3f}s"
            )
            logger.info(f"  Total Processed: {len(texts):,} texts")
            logger.info(f"  Quantization Time: {quantize_time:.3f}s (for all {len(texts):,} texts)")
            logger.info(f"  Cosine Similarity: {sim:.4f}")
        
        gc.collect()
        torch.mps.empty_cache()
        
        return combined_quantized, all_batch_ids, all_metrics


# --------------------------------------------------------------------------------------
# DB Updating with Retry
# --------------------------------------------------------------------------------------
async def update_embeddings(pool, batch_ids: List[int], embeddings: np.ndarray, retries: int = 3) -> Tuple[float, int]:
    for attempt in range(retries):
        try:
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
                            UPDATE farcaster.casts AS t
                            SET 
                                embedding384 = v.embedding::vector,
                                embedding384_updated_at = NOW()
                            FROM (VALUES {values_clause}) AS v(id, embedding)
                            WHERE t.id = v.id
                            RETURNING t.id
                        """
                        result = await conn.fetch(update_sql, timeout=DB_TIMEOUT)
                        rows_updated = len(result)
                        total_updated += rows_updated
                        
                        if rows_updated != len(chunk_ids):
                            logger.warning(
                                f"Warning: Expected to update {len(chunk_ids)} rows but updated {rows_updated}"
                            )
            
            return (time.time() - start_time, total_updated)
        
        except Exception as e:
            logger.warning(f"Update failed (attempt {attempt+1}/{retries}): {e}")
            if attempt == retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)
    
    # Fallback
    return 0.0, 0


# --------------------------------------------------------------------------------------
# Main Routine
# --------------------------------------------------------------------------------------
async def db_update_worker(pool, queue: asyncio.Queue, stats: ProcessingStats, update_event: asyncio.Event):
    last_log = time.time()
    updates_since_log = 0
    total_update_time = 0.0
    
    while True:
        item = await queue.get()
        if item is None:
            break
            
        batch_ids, embeddings = item
        start_time = time.time()
        try:
            update_time, rows_updated = await update_embeddings(pool, batch_ids, embeddings)
            total_update_time += update_time
            updates_since_log += rows_updated
            
            now = time.time()
            if now - last_log >= 5:
                avg_time = total_update_time / (updates_since_log or 1)
                logger.info("DB Update Stats:")
                logger.info(f"  Queue Size: {queue.qsize():,} batches")
                logger.info(f"  Recent Updates: {updates_since_log:,} rows")
                logger.info(f"  Avg Update Time: {avg_time:.3f}s per batch")
                logger.info(f"  Total Update Time: {total_update_time:.1f}s")
                logger.info(f"  Recent Batch Time: {update_time:.3f}s")
                
                last_log = now
                updates_since_log = 0
                total_update_time = 0
            
            stats.update(rows_updated)
        except Exception as e:
            # Critical error - propagate it by setting the event with an error
            logger.error(f"FATAL ERROR: Database update failed after all retries: {e}")
            # Set event so main process doesn't hang
            update_event.set()
            # Raise the exception to terminate the worker
            raise
        finally:
            # Always notify the main process we're done with this batch
            update_event.set()
            queue.task_done()

async def process_casts(pool, batch_size: int):
    stats = ProcessingStats()
    stats.total_remaining = await get_unprocessed_estimate(pool)
    logger.info(f"Starting processing of {stats.total_remaining:,} rows...")
    
    if stats.total_remaining == 0:
        logger.info("✨ No rows to process!")
        return
    
    processor = BatchProcessor(pool, BATCH_SIZE_ROWS)
    await processor.start()
    
    db_queue = asyncio.Queue()
    db_update_event = asyncio.Event()
    update_worker = asyncio.create_task(db_update_worker(pool, db_queue, stats, db_update_event))
    
    embedding_processor = EmbeddingProcessor(batch_size, quiet=True)
    batch_stats = BatchStats()
    processed_rows = 0
    
    max_process_memory = 0
    
    try:
        while True:
            # Check if worker has crashed
            if update_worker.done():
                # Worker is done, which means it either completed or raised an exception
                if update_worker.exception():
                    # Propagate the exception
                    logger.error("DB worker task failed with an exception. Terminating process.")
                    # Re-raise the exception to terminate main process
                    raise update_worker.exception()
                else:
                    # Worker completed normally, which shouldn't happen during the loop
                    logger.error("DB worker task completed unexpectedly. Terminating process.")
                    raise RuntimeError("DB worker task completed unexpectedly")
                    
            batch_summary = BatchSummary()
            op_timing = OperationTiming()
            
            op_timing.fetch_start = time.time() - op_timing.operation_start
            fetch_start = time.time()
            result = await processor.get_next_batch()
            if result is None:
                logger.info("✨ Processing completed successfully - no more rows to process!")
                break
                
            batch, fetch_time = result
            batch_summary.fetch_time = time.time() - fetch_start
            op_timing.fetch_end = time.time() - op_timing.operation_start
            
            if not batch:
                logger.info("✨ Processing completed successfully - no more rows to process!")
                break
            
            batch_summary.batch_size = len(batch)
            texts = [cast['text'] for cast in batch]
            ids = [cast['id'] for cast in batch]
            
            op_timing.inference_start = time.time() - op_timing.operation_start
            embed_start = time.time()
            
            process_mem_before = get_process_memory(os.getpid())
            logger.info(f"Memory before embedding: Process={process_mem_before:.1f}MB")
            
            embeddings, batch_ids, metrics = embedding_processor.process_batch(texts, ids)
            
            process_mem_after = get_process_memory(os.getpid())
            max_process_memory = max(max_process_memory, process_mem_after)
            logger.info(f"Memory after embedding: Process={process_mem_after:.1f}MB")
            logger.info(f"Memory change: Process={process_mem_after - process_mem_before:.1f}MB")
            
            batch_summary.embedding_time = time.time() - embed_start
            op_timing.inference_end = time.time() - op_timing.operation_start
            
            batch_summary.tokenization_time = sum(m.tokenization_time for m in metrics)
            batch_summary.forward_time = sum(m.model_forward_time for m in metrics)
            batch_summary.pooling_time = sum(m.pooling_time for m in metrics)
            batch_summary.quantization_time = sum(m.quantization_time for m in metrics)
            batch_summary.avg_cosine_sim = np.mean([m.cosine_sim for m in metrics]) if metrics else 0
            
            batch_stats.inference_metrics.extend(metrics)
            
            db_update_event.clear()
            op_timing.db_update_start = time.time() - op_timing.operation_start
            db_start = time.time()
            await db_queue.put((batch_ids, embeddings))
            await db_update_event.wait()
            
            # Check again if worker failed during this batch
            if update_worker.done() and update_worker.exception():
                logger.error("DB worker failed while processing the current batch")
                raise update_worker.exception()
                
            batch_summary.db_update_time = time.time() - db_start
            op_timing.db_update_end = time.time() - op_timing.operation_start
            
            processed_rows += len(batch_ids)
            batch_stats.update_processed(len(batch_ids))
            
            batch_summary.log_summary()
            op_timing.log_overlap()
            
            current_batch = processed_rows // batch_size
            total_batches = stats.total_remaining // batch_size
            batch_stats.log_quartile(current_batch, total_batches, processed_rows, stats.total_remaining)
    
    finally:
        # Cancel worker if it's still running
        if not update_worker.done():
            logger.info("Shutting down DB worker...")
            # Send None to signal worker to stop gracefully
            await db_queue.put(None)
            # Give worker time to complete current task
            try:
                await asyncio.wait_for(update_worker, timeout=10)
            except asyncio.TimeoutError:
                logger.warning("DB worker did not shut down cleanly, cancelling task")
                update_worker.cancel()
        
        # Check if worker encountered an exception
        try:
            # This will re-raise any exception from the worker
            await update_worker
        except Exception as e:
            logger.error(f"DB worker terminated with an error: {e}")
            # Re-raise to ensure main process exits with error
            raise
            
        total_time = time.time() - batch_stats.start_time
        logger.info("Processing complete!")
        logger.info(f"Total processed: {batch_stats.total_processed:,} rows")
        logger.info(f"Overall TPS: {batch_stats.total_processed / total_time:,.1f}")
        logger.info(f"Total time: {total_time/60:.1f} minutes")
        logger.info(f"Max memory usage: Process={max_process_memory:.1f}MB")


# --------------------------------------------------------------------------------------
# Main Entry Point
# --------------------------------------------------------------------------------------
async def main():
    tracemalloc.start()
    try:
        start_time = time.time()
        logger.info("Starting casts processing on Apple Silicon with MPS...")
        
        logger.info("Initial Memory Usage:")
        memory_info = detailed_memory_usage(tracemalloc_enabled=True)
        logger.info(f"  RSS: {memory_info['rss']:.1f} MB")
        logger.info(f"  VMS: {memory_info['vms']:.1f} MB")
        
        await db.initialize_pool(command_timeout=DB_TIMEOUT)
        await db.run_migrations(module_path=str(Path(__file__).parent))
        
        # Reset stale rows (optional safety net)
        reset_count = await reset_stale_rows(db.pool)
        logger.info(f"Reset {reset_count:,} stale rows")
        
        # Clear MPS cache if available
        try:
            torch.mps.empty_cache()
        except Exception as e:
            logger.warning(f"Could not configure MPS memory: {e}")
        
        await process_casts(db.pool, BATCH_SIZE)
        
        total_duration = time.time() - start_time
        logger.info(f"Total processing time: {total_duration:.1f}s")
        
        logger.info("Final Memory Usage:")
        memory_info = detailed_memory_usage(tracemalloc_enabled=True)
        logger.info(f"  RSS: {memory_info['rss']:.1f} MB")
        logger.info(f"  VMS: {memory_info['vms']:.1f} MB")
        
        if memory_info['tracemalloc']:
            logger.info("Top Memory Allocations:")
            for stat in memory_info['tracemalloc']:
                logger.info(f"  {stat.size/1024**2:.1f} MB: {stat.traceback.format()[0]}")
        
    finally:
        await db.close_pool()
        tracemalloc.stop()


if __name__ == "__main__":
    # Use 'spawn' start method on macOS
    multiprocessing.set_start_method('spawn')
    asyncio.run(main())
