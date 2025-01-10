"""
Comprehensive benchmark suite for embedding service performance.
Tests both single-instance and multi-instance configurations.
"""

import asyncio
import time
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import psutil
from dataclasses import dataclass
import aiohttp
import argparse

from src.embedding.scaling import ScalingManager
from src.db.queries import fetch_unprocessed_casts

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    """Stores benchmark results."""
    texts_per_second: float
    memory_mb: float
    batch_size: int
    total_texts: int
    duration_seconds: float
    num_workers: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "texts_per_second": round(self.texts_per_second, 2),
            "memory_mb": round(self.memory_mb, 2),
            "batch_size": self.batch_size,
            "total_texts": self.total_texts,
            "duration_seconds": round(self.duration_seconds, 2),
            "num_workers": self.num_workers
        }

async def benchmark_single_instance(texts: List[str], batch_size: int = 128) -> BenchmarkResult:
    """Benchmark single instance performance."""
    logger.info(f"\nTesting single instance with batch_size={batch_size}")
    logger.info("=" * 80)
    
    start_time = time.time()
    process = psutil.Process()
    start_memory = process.memory_info().rss / (1024 * 1024)
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://localhost:8374/embed384",
            json={"texts": texts, "batch_size": batch_size}
        ) as response:
            result = await response.json()
            
    duration = time.time() - start_time
    end_memory = process.memory_info().rss / (1024 * 1024)
    
    return BenchmarkResult(
        texts_per_second=len(texts) / duration,
        memory_mb=end_memory - start_memory,
        batch_size=batch_size,
        total_texts=len(texts),
        duration_seconds=duration,
        num_workers=1
    )

async def benchmark_multi_instance(
    texts: List[str],
    num_workers: int,
    batch_size: int = 128
) -> BenchmarkResult:
    """Benchmark multi-instance performance."""
    logger.info(f"\nTesting {num_workers} instances with batch_size={batch_size}")
    logger.info("=" * 80)
    
    start_time = time.time()
    process = psutil.Process()
    start_memory = process.memory_info().rss / (1024 * 1024)
    
    # Initialize scaling manager
    manager = ScalingManager(max_workers=num_workers)
    
    # Process texts
    embeddings = await manager.process_texts(texts)
    
    duration = time.time() - start_time
    end_memory = process.memory_info().rss / (1024 * 1024)
    
    return BenchmarkResult(
        texts_per_second=len(texts) / duration,
        memory_mb=end_memory - start_memory,
        batch_size=batch_size,
        total_texts=len(texts),
        duration_seconds=duration,
        num_workers=num_workers
    )

async def run_comprehensive_benchmark(
    num_texts: int = 100000,
    batch_sizes: List[int] = [128, 256, 512],
    worker_counts: List[int] = [1, 3, 6, 9]
) -> List[Dict[str, Any]]:
    """Run comprehensive benchmarks across configurations."""
    results = []
    
    # Fetch test texts
    texts = [cast['text'] for cast in fetch_unprocessed_casts(batch_size=num_texts)]
    logger.info(f"Running benchmarks with {len(texts)} texts")
    
    # Single instance tests
    for batch_size in batch_sizes:
        result = await benchmark_single_instance(texts[:10000], batch_size)
        results.append(result.to_dict())
        logger.info(f"Single instance result: {result.texts_per_second:.1f} texts/sec")
        await asyncio.sleep(5)  # Cool down
        
    # Multi-instance tests
    for num_workers in worker_counts[1:]:  # Skip 1 worker as it's already tested
        for batch_size in batch_sizes:
            result = await benchmark_multi_instance(texts, num_workers, batch_size)
            results.append(result.to_dict())
            logger.info(
                f"{num_workers} workers result: {result.texts_per_second:.1f} texts/sec"
            )
            await asyncio.sleep(10)  # Longer cool down for multi-instance
            
    return results

def save_results(results: List[Dict[str, Any]]) -> None:
    """Save benchmark results to file."""
    Path("results").mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    result_file = f"results/benchmark_{timestamp}.json"
    
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)
        
    logger.info(f"\nResults saved to {result_file}")
    
def print_summary(results: List[Dict[str, Any]]) -> None:
    """Print summary of benchmark results."""
    logger.info("\nBenchmark Summary:")
    logger.info("=" * 80)
    
    # Single instance results
    single_results = [r for r in results if r['num_workers'] == 1]
    best_single = max(single_results, key=lambda x: x['texts_per_second'])
    logger.info(
        f"Best single instance: {best_single['texts_per_second']:.1f} texts/sec "
        f"(batch_size={best_single['batch_size']})"
    )
    
    # Multi-instance results
    multi_results = [r for r in results if r['num_workers'] > 1]
    if multi_results:
        best_multi = max(multi_results, key=lambda x: x['texts_per_second'])
        logger.info(
            f"Best multi-instance: {best_multi['texts_per_second']:.1f} texts/sec "
            f"({best_multi['num_workers']} workers, batch_size={best_multi['batch_size']})"
        )
    
    logger.info("=" * 80)

async def main():
    parser = argparse.ArgumentParser(description="Benchmark embedding service")
    parser.add_argument("--texts", type=int, default=100000,
                       help="Number of texts to process")
    parser.add_argument("--batch-sizes", type=int, nargs="+",
                       default=[128, 256, 512],
                       help="Batch sizes to test")
    parser.add_argument("--workers", type=int, nargs="+",
                       default=[1, 3, 6, 9],
                       help="Number of workers to test")
    args = parser.parse_args()
    
    try:
        results = await run_comprehensive_benchmark(
            num_texts=args.texts,
            batch_sizes=args.batch_sizes,
            worker_counts=args.workers
        )
        save_results(results)
        print_summary(results)
        
    except Exception as e:
        logger.error(f"Benchmark failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 