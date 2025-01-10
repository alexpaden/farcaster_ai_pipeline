"""
Benchmark utilities for the embedding384 module.
Provides tools for testing different batch sizes and instance configurations.
"""

import os
import warnings
import psutil
import time
import json
import asyncio
import asyncpg
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import gc
import multiprocessing
from transformers import AutoTokenizer, AutoModel
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure warnings
warnings.filterwarnings("ignore", message=".*_is_quantized_training_enabled.*")
warnings.filterwarnings("ignore", message=".*loss_type=None.*")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@dataclass
class BenchmarkMetrics:
    """Stores benchmark results."""
    texts_per_second: float
    memory_usage_mb: float
    batch_size: int
    total_texts: int
    duration_seconds: float
    num_instances: int
    timestamp: datetime = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "texts_per_second": round(self.texts_per_second, 2),
            "memory_usage_mb": round(self.memory_usage_mb, 2),
            "batch_size": self.batch_size,
            "total_texts": self.total_texts,
            "duration_seconds": round(self.duration_seconds, 2),
            "num_instances": self.num_instances,
            "timestamp": self.timestamp.isoformat()
        }

class OptimizedEmbeddingModel:
    """Optimized model for benchmarking different configurations."""
    
    def __init__(self, batch_size: int):
        self.device = torch.device("mps")
        self.batch_size = batch_size
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        
        logger.info(f"Initializing model with batch_size={batch_size} on MPS...")
        self._initialize_model()
        
    def _initialize_model(self):
        # Use all CPU cores
        torch.set_num_threads(multiprocessing.cpu_count())
        gc.collect()
        
        # Load model in float16 for better performance
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        base_model = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        base_model.to(self.device)
        base_model.eval()
        
        # Initialize with longer sample text for buffer sizing
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
        """Warm up the model with a small batch."""
        logger.info("Warming up model...")
        warmup_texts = ["Warm-up sentence"] * 8
        _ = self.encode(warmup_texts)
        logger.info("Warm-up complete")
        
    def _mean_pooling(self, model_output, attention_mask):
        """Mean pooling of token embeddings."""
        token_embeddings = model_output['last_hidden_state']
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        normalized = sum_embeddings / sum_mask
        return torch.nn.functional.normalize(normalized, p=2, dim=1)
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings using pre-allocated buffers."""
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

def get_process_memory(pid: int) -> float:
    """Get memory usage for a process and all its children in MB."""
    try:
        parent = psutil.Process(pid)
        memory = parent.memory_info().rss
        for child in parent.children(recursive=True):
            try:
                memory += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return memory / (1024 * 1024)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0

async def run_benchmark(
    texts: List[str],
    batch_size: int,
    num_instances: int = 1,
    pool: Optional[asyncpg.Pool] = None
) -> BenchmarkMetrics:
    """Run benchmark with specified configuration."""
    logger.info(f"Running benchmark with batch_size={batch_size}, instances={num_instances}")
    
    if num_instances == 1:
        return await _run_single_instance(texts, batch_size)
    else:
        return await _run_multi_instance(texts, batch_size, num_instances)

async def _run_single_instance(texts: List[str], batch_size: int) -> BenchmarkMetrics:
    """Run benchmark with a single instance."""
    model = OptimizedEmbeddingModel(batch_size)
    start_time = time.time()
    pid = os.getpid()
    peak_memory = 0
    
    # Process texts in batches
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        _ = model.encode(batch)
        
        # Track peak memory
        current_memory = get_process_memory(pid)
        peak_memory = max(peak_memory, current_memory)
    
    duration = time.time() - start_time
    
    return BenchmarkMetrics(
        texts_per_second=len(texts) / duration,
        memory_usage_mb=peak_memory,
        batch_size=batch_size,
        total_texts=len(texts),
        duration_seconds=duration,
        num_instances=1
    )

async def _run_multi_instance(
    texts: List[str],
    batch_size: int,
    num_instances: int
) -> BenchmarkMetrics:
    """Run benchmark with multiple instances."""
    # Split texts among instances
    texts_per_instance = len(texts) // num_instances
    queue = multiprocessing.Queue()
    processes = []
    
    # Start processes
    for i in range(num_instances):
        start_idx = i * texts_per_instance
        end_idx = start_idx + texts_per_instance
        instance_texts = texts[start_idx:end_idx]
        
        p = multiprocessing.Process(
            target=_run_instance_process,
            args=(instance_texts, batch_size, queue)
        )
        p.start()
        processes.append(p)
    
    # Wait for initialization
    await asyncio.sleep(10)
    
    # Measure peak memory across all processes
    total_peak_memory = 0
    for _ in range(5):  # Multiple measurements
        current_total = sum(
            get_process_memory(p.pid)
            for p in processes
            if p.is_alive()
        )
        total_peak_memory = max(total_peak_memory, current_total)
        await asyncio.sleep(2)
    
    # Wait for completion
    for p in processes:
        p.join()
    
    # Collect metrics
    instance_metrics = []
    while not queue.empty():
        metrics = queue.get()
        instance_metrics.append(metrics)
    
    # Aggregate metrics
    total_texts = sum(m.total_texts for m in instance_metrics)
    total_duration = max(m.duration_seconds for m in instance_metrics)
    avg_memory_per_instance = total_peak_memory / num_instances
    
    return BenchmarkMetrics(
        texts_per_second=total_texts / total_duration,
        memory_usage_mb=avg_memory_per_instance,
        batch_size=batch_size,
        total_texts=total_texts,
        duration_seconds=total_duration,
        num_instances=num_instances
    )

def _run_instance_process(texts: List[str], batch_size: int, queue: multiprocessing.Queue):
    """Process function for multi-instance benchmarking."""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        metrics = loop.run_until_complete(_run_single_instance(texts, batch_size))
        queue.put(metrics)
    except Exception as e:
        logger.error(f"Instance process error: {str(e)}")
        raise 