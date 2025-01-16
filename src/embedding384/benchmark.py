"""
Benchmark embedding generation using cast texts.
Tests performance across different batch sizes (128, 256, 512, 1024) with both single and multi-instance scaling.
Uses TorchScript optimization and float16 precision on MPS, with int8 quantization for storage.
"""

# Configuration settings
BATCH_SIZE = 256  # Optimal batch size for MPS
NUM_PROCESSES = 12  # Number of parallel processes
TEST_SAMPLE_SIZE = 100000  # Number of test casts to sample
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_SEQ_LENGTH = 256  # Maximum sequence length for tokenization (model max is 256)

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
from dataclasses import dataclass
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

@dataclass
class BenchmarkMetrics:
    texts_per_second: float
    memory_usage_mb: float
    batch_size: int
    total_texts: int
    duration_seconds: float
    num_instances: int
    avg_cosine_sim: float = 0.0  # Average cosine similarity between float16 and int8
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "texts_per_second": round(self.texts_per_second, 2),
            "memory_usage_mb": round(self.memory_usage_mb, 2),
            "batch_size": self.batch_size,
            "total_texts": self.total_texts,
            "duration_seconds": round(self.duration_seconds, 2),
            "num_instances": self.num_instances,
            "avg_cosine_sim": round(self.avg_cosine_sim, 6)
        }

def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute cosine similarity between two tensors."""
    return float(torch.nn.functional.cosine_similarity(a, b, dim=1).mean().item())

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

async def get_table_estimate(pool, table_name: str) -> int:
    """Get fast row count estimate using pg_class statistics."""
    async with pool.acquire() as conn:
        result = await conn.fetchval("""
            SELECT reltuples::bigint AS estimate
            FROM pg_class
            WHERE relname = $1
        """, table_name)
        return int(result or 0)

async def reset_test_casts(pool):
    """Reset and populate test_casts table with sample casts."""
    async with pool.acquire() as conn:
        await conn.execute(f"""
            TRUNCATE TABLE public.test_casts;
            
            INSERT INTO public.test_casts (id, text)
            SELECT id, text
            FROM public.casts
            WHERE text IS NOT NULL 
            AND length(trim(text)) > 0
            ORDER BY id
            LIMIT {TEST_SAMPLE_SIZE};
        """)
        # Update table statistics
        await conn.execute("ANALYZE public.test_casts")

async def fetch_test_batch(pool, limit: int = 100) -> List[Dict[str, Any]]:
    """Fetch a batch of test casts."""
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT id, text 
            FROM public.test_casts 
            WHERE embedding384 IS NULL
            ORDER BY id
            LIMIT $1
        """, limit)
        return [dict(row) for row in rows]

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

def run_benchmark_process(texts: List[str], ids: List[int], batch_size: int, progress_queue=None, metrics_queue=None):
    """Run benchmark in a separate process."""
    model = OptimizedEmbeddingModel(batch_size, quiet=True)
    
    start_time = time.time()
    pid = os.getpid()
    peak_memory = 0
    total_batches = len(range(0, len(texts), batch_size))
    total_sim = 0.0
    batch_count = 0
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(db.initialize_pool())
        
        async def process_batches():
            nonlocal total_sim, batch_count
            async with db.pool.acquire() as conn:
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
                            UPDATE test_casts AS t
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
        print(f"Error in benchmark process: {str(e)}")
        raise

async def run_benchmark(pool, batch_size: int, num_instances: int):
    """Run benchmark with specified batch size and number of instances."""
    print(f"\nRunning benchmark with batch_size={batch_size}, instances={num_instances}")
    
    # Fetch test data
    all_casts = await fetch_test_batch(pool, 100000)
    texts = [cast['text'] for cast in all_casts]
    ids = [cast['id'] for cast in all_casts]
    print(f"Processing {len(texts)} texts...")
    
    if num_instances == 1:
        # Single instance benchmark
        total_batches = len(range(0, len(texts), batch_size))
        with tqdm(total=total_batches, desc="Processing", unit="batch") as pbar:
            metrics = run_benchmark_process(texts, ids, batch_size)
            pbar.update(total_batches)
        return metrics
    else:
        # Multi-instance benchmark
        texts_per_instance = len(texts) // num_instances
        progress_queue = multiprocessing.Queue()
        metrics_queue = multiprocessing.Queue()
        processes = []
        
        # Calculate total batches across all instances
        total_batches = sum(len(range(0, texts_per_instance, batch_size)) for _ in range(num_instances))
        
        # Start processes
        for i in range(num_instances):
            start_idx = i * texts_per_instance
            end_idx = start_idx + texts_per_instance
            instance_texts = texts[start_idx:end_idx]
            instance_ids = ids[start_idx:end_idx]
            
            p = multiprocessing.Process(
                target=run_benchmark_process,
                args=(instance_texts, instance_ids, batch_size, progress_queue, metrics_queue)
            )
            p.start()
            processes.append(p)
        
        # Monitor progress with a single progress bar
        with tqdm(total=total_batches, desc="Processing", unit="batch") as pbar:
            completed_batches = 0
            while completed_batches < total_batches:
                try:
                    progress = progress_queue.get(timeout=1)
                    completed_batches += progress
                    pbar.update(progress)
                except:
                    # Check if processes are still alive
                    if not any(p.is_alive() for p in processes):
                        break
        
        # Wait for completion and collect metrics
        for p in processes:
            p.join()
        
        instance_metrics = []
        while not metrics_queue.empty():
            metrics = metrics_queue.get()
            instance_metrics.append(metrics)
        
        # Aggregate metrics
        total_texts = sum(m.total_texts for m in instance_metrics)
        total_duration = max(m.duration_seconds for m in instance_metrics)
        total_peak_memory = sum(m.memory_usage_mb for m in instance_metrics)
        avg_sim = sum(m.avg_cosine_sim for m in instance_metrics) / len(instance_metrics)
        
        metrics = BenchmarkMetrics(
            texts_per_second=total_texts / total_duration,
            memory_usage_mb=total_peak_memory / num_instances,
            batch_size=batch_size,
            total_texts=total_texts,
            duration_seconds=total_duration,
            num_instances=num_instances,
            avg_cosine_sim=avg_sim
        )
        
        print(f"\nTotal peak memory: {total_peak_memory:.1f}MB")
        return metrics

async def main():
    """Main benchmark orchestration."""
    try:
        start_time = time.time()
        print("\nStarting benchmark...")
        
        await db.initialize_pool()
        await db.run_migrations(module_path=str(Path(__file__).parent))
        
        results = []
        batch_sizes = [BATCH_SIZE]  # Using configured batch size
        instance_counts = [NUM_PROCESSES]  # Using configured process count
        
        for batch_size in batch_sizes:
            print("\nPreparing test data...")
            await reset_test_casts(db.pool)
            
            # Get row estimate after population
            total_rows = await get_table_estimate(db.pool, "test_casts")
            print(f"Estimated rows in test table: {total_rows:,}")
            
            for num_instances in instance_counts:
                metrics = await run_benchmark(db.pool, batch_size, num_instances)
                results.append({
                    "batch_size": batch_size,
                    "num_instances": num_instances,
                    **metrics.to_dict()
                })
                
                gc.collect()
        
        # Save results
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(exist_ok=True)
        result_file = results_dir / f"benchmark_{time.strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        total_duration = time.time() - start_time
        
        print(f"\nBenchmark results saved to {result_file}")
        print("\nResults summary:")
        print("=" * 80)
        for result in results:
            total_memory = result['memory_usage_mb'] * result['num_instances']
            print(f"Batch size: {result['batch_size']}, "
                  f"Instances: {result['num_instances']}, "
                  f"TPS: {result['texts_per_second']:.1f}, "
                  f"Memory/Instance: {result['memory_usage_mb']:.1f}MB, "
                  f"Total Memory: {total_memory:.1f}MB, "
                  f"Avg Cosine Sim: {result['avg_cosine_sim']:.6f}")
        print(f"Total time: {total_duration:.1f}s")
        print("=" * 80)
        
    finally:
        await db.close_pool()

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    asyncio.run(main()) 