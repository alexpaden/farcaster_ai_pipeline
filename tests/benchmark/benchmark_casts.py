"""
Benchmark embedding generation using cast texts.
Tests performance across different batch sizes (128, 256, 512, 1024) with both single and multi-instance scaling.
Uses TorchScript optimization and float16 precision on MPS.
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
import asyncpg
from typing import List, Dict, Any
from dataclasses import dataclass
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm
import torch
import torch.nn as nn
import gc
import multiprocessing
from transformers import AutoTokenizer, AutoModel

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
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "texts_per_second": round(self.texts_per_second, 2),
            "memory_usage_mb": round(self.memory_usage_mb, 2),
            "batch_size": self.batch_size,
            "total_texts": self.total_texts,
            "duration_seconds": round(self.duration_seconds, 2),
            "num_instances": self.num_instances
        }

class OptimizedEmbeddingModel:
    def __init__(self, batch_size: int):
        self.device = torch.device("mps")
        self.batch_size = batch_size
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        
        # Initialize model with optimizations
        print(f"\nInitializing model with batch_size={batch_size} on MPS...")
        self._initialize_model()
        
    def _initialize_model(self):
        # Use all CPU cores
        torch.set_num_threads(multiprocessing.cpu_count())
        
        # Force garbage collection
        gc.collect()
        
        # Load model in float16
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        base_model = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        base_model.to(self.device)
        base_model.eval()
        
        # Initialize with longer sample text
        sample_texts = ["This is a longer initialization text that will ensure adequate buffer sizes"] * 32
        inputs = self.tokenizer(
            sample_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(self.device)
        
        # TorchScript optimization
        with torch.inference_mode():
            traced_model = torch.jit.trace(
                base_model,
                (inputs['input_ids'], inputs['attention_mask']),
                strict=False
            )
            self.model = torch.jit.optimize_for_inference(traced_model)
        
        # Pre-allocate buffers
        self.input_buffers = {
            'input_ids': torch.zeros((self.batch_size, 128), dtype=torch.long, device=self.device),
            'attention_mask': torch.zeros((self.batch_size, 128), dtype=torch.long, device=self.device)
        }
        
        # Get output shape and pre-allocate output buffer
        with torch.inference_mode():
            outputs = self.model(inputs['input_ids'], inputs['attention_mask'])
            embeddings = self._mean_pooling(outputs, inputs['attention_mask'])
            self.output_buffer = torch.zeros(
                (self.batch_size, embeddings.shape[1]),
                dtype=torch.float16,
                device=self.device
            )
        
        # Cleanup
        del base_model, traced_model, outputs, embeddings
        gc.collect()
        
        # Warm up
        self._warm_up()
        
    def _warm_up(self):
        print("Warming up model...")
        warmup_texts = ["Warm-up sentence"] * 8
        _ = self.encode(warmup_texts)
        print("Warm-up complete")
        
    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output['last_hidden_state']
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        normalized = sum_embeddings / sum_mask
        return torch.nn.functional.normalize(normalized, p=2, dim=1)
    
    def encode(self, texts: List[str]) -> np.ndarray:
        with torch.inference_mode():
            # Tokenize
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )
            
            # Use pre-allocated buffers
            for k, v in inputs.items():
                if k in self.input_buffers:
                    self.input_buffers[k][:v.size(0), :v.size(1)] = v.to(self.device)
                    inputs[k] = self.input_buffers[k][:v.size(0), :v.size(1)]
            
            # Forward pass
            outputs = self.model(inputs['input_ids'], inputs['attention_mask'])
            embeddings = self._mean_pooling(outputs, inputs['attention_mask'])
            
            # Use output buffer
            self.output_buffer[:embeddings.size(0)] = embeddings
            return self.output_buffer[:embeddings.size(0)].cpu().numpy()

async def reset_test_casts(pool):
    """Reset and populate test_casts table with 100k casts."""
    print("\nResetting test_casts table...")
    async with pool.acquire() as conn:
        await conn.execute("""
            TRUNCATE TABLE public.test_casts;
            
            INSERT INTO public.test_casts (cast_id, text)
            SELECT id, text
            FROM public.casts
            WHERE text IS NOT NULL 
            AND length(trim(text)) > 0
            ORDER BY id
            LIMIT 100000;
        """)
    print("Test casts table reset complete.")

async def fetch_test_batch(pool, limit: int = 100) -> List[Dict[str, Any]]:
    """Fetch a batch of test casts."""
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT cast_id, text 
            FROM public.test_casts 
            ORDER BY cast_id
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

def run_benchmark_process(texts: List[str], batch_size: int, queue=None):
    """Run benchmark in a separate process."""
    model = OptimizedEmbeddingModel(batch_size)
    
    start_time = time.time()
    pid = os.getpid()
    peak_memory = 0
    
    # Process texts in batches
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        _ = model.encode(batch)
        
        # Track peak memory (both RAM and GPU)
        current_memory = get_process_memory(pid) + get_gpu_memory()
        peak_memory = max(peak_memory, current_memory)
    
    duration = time.time() - start_time
    
    metrics = BenchmarkMetrics(
        texts_per_second=len(texts) / duration,
        memory_usage_mb=peak_memory,
        batch_size=batch_size,
        total_texts=len(texts),
        duration_seconds=duration,
        num_instances=1
    )
    
    if queue:
        queue.put(metrics)
    return metrics

async def run_benchmark(pool, batch_size: int, num_instances: int):
    """Run benchmark with specified batch size and number of instances."""
    print(f"\nRunning benchmark with batch_size={batch_size}, instances={num_instances}")
    
    # Fetch test casts
    all_casts = await fetch_test_batch(pool, 100000)
    texts = [cast['text'] for cast in all_casts]
    
    if num_instances == 1:
        # Single instance benchmark
        metrics = run_benchmark_process(texts, batch_size)
        print(f"Single instance results: {metrics.to_dict()}")
        return metrics
    else:
        # Multi-instance benchmark
        texts_per_instance = len(texts) // num_instances
        queue = multiprocessing.Queue()
        processes = []
        
        for i in range(num_instances):
            start_idx = i * texts_per_instance
            end_idx = start_idx + texts_per_instance
            instance_texts = texts[start_idx:end_idx]
            
            p = multiprocessing.Process(
                target=run_benchmark_process,
                args=(instance_texts, batch_size, queue)
            )
            p.start()
            processes.append(p)
        
        # Wait longer for processes to fully initialize
        time.sleep(10)  # Increased from 2s to 10s
        
        # Measure peak memory across all processes
        total_peak_memory = 0
        measurement_attempts = 5
        
        # Take multiple measurements to catch peak usage
        for _ in range(measurement_attempts):
            current_total = 0
            for p in processes:
                try:
                    process_memory = get_process_memory(p.pid) + get_gpu_memory()
                    current_total += process_memory
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            total_peak_memory = max(total_peak_memory, current_total)
            time.sleep(2)  # Wait between measurements
        
        # Wait for completion
        for p in processes:
            p.join()
        
        # Collect performance metrics
        instance_metrics = []
        while not queue.empty():
            metrics = queue.get()
            instance_metrics.append(metrics)
        
        # Aggregate metrics
        total_texts = sum(m.total_texts for m in instance_metrics)
        total_duration = max(m.duration_seconds for m in instance_metrics)
        avg_memory_per_instance = total_peak_memory / num_instances if num_instances > 0 else 0
        
        metrics = BenchmarkMetrics(
            texts_per_second=total_texts / total_duration,
            memory_usage_mb=avg_memory_per_instance,
            batch_size=batch_size,
            total_texts=total_texts,
            duration_seconds=total_duration,
            num_instances=num_instances
        )
        
        print(f"Multi-instance results: {metrics.to_dict()}\n")
        print(f"Total peak memory across all instances: {total_peak_memory:.1f}MB")
        return metrics

async def main():
    """Main benchmark orchestration."""
    # Connect to database
    pool = await asyncpg.create_pool(
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
        database=os.getenv('DB_NAME'),
        host=os.getenv('DB_HOST'),
        port=int(os.getenv('DB_PORT'))
    )
    
    try:
        results = []
        batch_sizes = [256]  # Fixed optimal batch size
        instance_counts = [1, 6, 36]  # Testing key scaling points
        
        for batch_size in batch_sizes:
            # Reset test data
            await reset_test_casts(pool)
            
            for num_instances in instance_counts:
                print(f"\nTesting with {num_instances} instances")
                metrics = await run_benchmark(pool, batch_size, num_instances)
                results.append({
                    "batch_size": batch_size,
                    "num_instances": num_instances,
                    **metrics.to_dict()
                })
                
                # Clean up
                gc.collect()
                await asyncio.sleep(5)
        
        # Save results
        Path("results").mkdir(exist_ok=True)
        result_file = f"results/benchmark_{time.strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"\nBenchmark results saved to {result_file}")
        print("\nResults summary:")
        print("=" * 80)
        for result in results:
            total_memory = result['memory_usage_mb'] * result['num_instances']
            print(f"Batch size: {result['batch_size']}, "
                  f"Instances: {result['num_instances']}, "
                  f"TPS: {result['texts_per_second']:.1f}, "
                  f"Memory/Instance: {result['memory_usage_mb']:.1f}MB, "
                  f"Total Memory: {total_memory:.1f}MB")
        print("=" * 80)
        
    finally:
        await pool.close()

if __name__ == "__main__":
    asyncio.run(main()) 