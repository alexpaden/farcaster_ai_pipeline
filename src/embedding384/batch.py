"""
Batch processing logic for the embedding pipeline.
"""

import gc
import torch
import numpy as np
import asyncio
from typing import List, Dict, Any, Tuple, Optional
from .logger import PipelineLogger, InferenceMetrics
from .db import fetch_next_batch, get_unprocessed_estimate
from .utils import log_and_reraise
from .model import OptimizedEmbeddingModel

class BatchProcessor:
    """Handles batch processing and prefetching."""
    def __init__(self, pool, batch_size: int, prefetch_batches: int = 3, logger: Optional[PipelineLogger] = None):
        self.pool = pool
        self.batch_size = batch_size
        self.prefetch_queue = asyncio.Queue(maxsize=prefetch_batches)
        self.last_id = 0
        self.done = False
        self._prefetch_task: Optional[asyncio.Task] = None
        self.logger = logger

    async def start(self):
        """Start the prefetch worker."""
        self._prefetch_task = asyncio.create_task(self.prefetch_worker())

    async def prefetch_worker(self):
        """Continuously prefetch next batches."""
        try:
            while not self.done:
                current_size = self.prefetch_queue.qsize()

                if current_size < self.prefetch_queue.maxsize:
                    batch, fetch_time = await fetch_next_batch(
                        self.pool,
                        self.last_id,
                        self.batch_size,
                        self.logger
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
            log_and_reraise(e, "prefetch worker")
            self.done = True

    async def get_next_batch(self) -> Optional[Tuple[List[Dict[str, Any]], float]]:
        """Get the next batch from the queue."""
        if self.done and self.prefetch_queue.empty():
            return None
        try:
            return await self.prefetch_queue.get()
        except asyncio.QueueEmpty:
            return None

class EmbeddingProcessor:
    """Handles model initialization and inference."""
    def __init__(self, model_batch_size: int, full_batch_size: int, quiet: bool = True, logger: Optional[PipelineLogger] = None):
        self.model = OptimizedEmbeddingModel(model_batch_size, quiet=quiet, logger=logger)
        self.model_batch_size = model_batch_size
        self.full_batch_size = full_batch_size
        self._processed_in_batch = 0
        self._current_metrics = []
    
    def process_batch(self, texts: List[str], ids: List[int]) -> Tuple[np.ndarray, List[int], List[InferenceMetrics]]:
        """Process a batch of texts."""
        all_embeddings = []
        all_batch_ids = []
        all_metrics = []
        
        # Process in model batch size chunks
        for i in range(0, len(texts), self.model_batch_size):
            batch_texts = texts[i:i + self.model_batch_size]
            batch_ids = ids[i:i + self.model_batch_size]
            
            # Process batch and collect metrics
            embeddings, sim, metrics = self.model.encode(batch_texts)
            
            all_embeddings.append(embeddings)
            all_batch_ids.extend(batch_ids)
            all_metrics.append(metrics)
            
            # Track progress across the full batch
            self._processed_in_batch += len(batch_texts)
            self._current_metrics.append(metrics)
            
            # Only log at quarter points of the full batch
            quarter_size = self.full_batch_size // 4
            if self._processed_in_batch % quarter_size < self.model_batch_size:
                quarter_number = self._processed_in_batch // quarter_size
                if quarter_number <= 4:  # Only log for the first 4 quarters
                    # Calculate average metrics over recent batches
                    recent_metrics = self._current_metrics[-min(10, len(self._current_metrics)):]
                    avg_metrics = InferenceMetrics(
                        tokenization_time=np.mean([m.tokenization_time for m in recent_metrics]),
                        model_forward_time=np.mean([m.model_forward_time for m in recent_metrics]),
                        pooling_time=np.mean([m.pooling_time for m in recent_metrics]),
                        quantization_time=np.mean([m.quantization_time for m in recent_metrics]),
                        total_time=np.mean([m.total_time for m in recent_metrics]),
                        batch_size=metrics.batch_size,
                        texts_per_second=np.mean([m.texts_per_second for m in recent_metrics])
                    )
                    
                    print(f"\nProcessing {self._processed_in_batch:,}/{self.full_batch_size:,} rows ({quarter_number}/4 of current batch)")
                    print(f"Recent metrics:")
                    print(f"  Tokenization: {avg_metrics.tokenization_time:.3f}s")
                    print(f"  Model Forward: {avg_metrics.model_forward_time:.3f}s")
                    print(f"  Pooling: {avg_metrics.pooling_time:.3f}s")
                    print(f"  Quantization: {avg_metrics.quantization_time:.3f}s")
                    print(f"  Total: {avg_metrics.total_time:.3f}s")
                    print(f"  Throughput: {avg_metrics.texts_per_second:.1f} texts/sec")
            
            # Reset counters if we've processed a full batch
            if self._processed_in_batch >= self.full_batch_size:
                self._processed_in_batch = 0
                self._current_metrics = []
            
            # Force garbage collection between batches
            gc.collect()
            torch.mps.empty_cache()
        
        # Combine all batches
        combined_embeddings = np.vstack(all_embeddings)
        return combined_embeddings, all_batch_ids, all_metrics

    def quantize_embeddings(self, embeddings: torch.Tensor) -> np.ndarray:
        """Optimized int8 quantization."""
        with torch.no_grad():
            # Process in chunks for better memory efficiency
            chunk_size = 8192
            results = []
            
            for i in range(0, len(embeddings), chunk_size):
                chunk = embeddings[i:i + chunk_size]
                # Scale to int8 range efficiently
                chunk = chunk.cpu()  # Move to CPU once
                chunk = ((chunk * 127.0).round().clip(-127, 127)).to(torch.int8)
                results.append(chunk.numpy())
            
            return np.vstack(results) 