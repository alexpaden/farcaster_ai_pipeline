"""
Manages multiple embedding service instances for parallel processing.
Automatically scales based on workload size and time constraints.
"""

import asyncio
import aiohttp
import subprocess
import time
import logging
from typing import List, Dict, Any
from dataclasses import dataclass
import math

logger = logging.getLogger(__name__)

@dataclass
class WorkerConfig:
    port: int
    batch_size: int = 128
    memory_limit_gb: int = 6
    texts_per_second: float = 400.0

class ScalingManager:
    def __init__(self, base_port: int = 8374, max_workers: int = 9):
        self.base_port = base_port
        self.max_workers = max_workers
        self.active_workers = 1  # Base worker always running
        self.worker_configs = [
            WorkerConfig(port=base_port + i)
            for i in range(max_workers)
        ]
        
    async def _check_worker_health(self, port: int) -> bool:
        """Check if a worker is healthy."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"http://localhost:{port}/health") as response:
                    if response.status == 200:
                        return True
        except:
            return False
        return False
        
    async def scale_workers(self, num_workers: int) -> None:
        """Scale to desired number of workers."""
        try:
            # Don't exceed max workers
            num_workers = min(num_workers, self.max_workers)
            
            if num_workers == self.active_workers:
                return
                
            logger.info(f"Scaling from {self.active_workers} to {num_workers} workers")
            
            # Start/stop workers via supervisord
            cmd = f"supervisorctl start embedding_{self.base_port + num_workers - 1}" if num_workers > self.active_workers else f"supervisorctl stop embedding_{self.base_port + self.active_workers - 1}"
            subprocess.run(cmd, shell=True, check=True)
            
            # Wait for workers to be healthy
            await asyncio.sleep(5)
            
            # Verify all workers are healthy
            tasks = []
            for i in range(num_workers):
                port = self.base_port + i
                tasks.append(self._check_worker_health(port))
                
            results = await asyncio.gather(*tasks)
            if not all(results):
                raise Exception("Not all workers are healthy after scaling")
                
            self.active_workers = num_workers
            logger.info(f"Successfully scaled to {num_workers} workers")
            
        except Exception as e:
            logger.error(f"Error scaling workers: {str(e)}")
            raise
            
    def calculate_needed_workers(self, num_texts: int, time_limit_minutes: float = 5.0) -> int:
        """Calculate number of workers needed to process texts within time limit."""
        texts_per_second_per_worker = self.worker_configs[0].texts_per_second
        total_seconds = time_limit_minutes * 60
        
        # Calculate workers needed to meet time limit
        needed_workers = math.ceil(
            num_texts / (texts_per_second_per_worker * total_seconds)
        )
        
        return min(needed_workers, self.max_workers)
        
    async def process_texts(self, texts: List[str], time_limit_minutes: float = 5.0) -> List[List[float]]:
        """Process texts with automatic scaling."""
        try:
            # Calculate needed workers
            needed_workers = self.calculate_needed_workers(len(texts), time_limit_minutes)
            
            # Scale if needed
            if needed_workers > self.active_workers:
                await self.scale_workers(needed_workers)
                
            # Split texts among workers
            chunk_size = math.ceil(len(texts) / self.active_workers)
            chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
            
            # Process chunks in parallel
            async with aiohttp.ClientSession() as session:
                tasks = []
                for i, chunk in enumerate(chunks):
                    port = self.base_port + i
                    task = asyncio.create_task(self._process_chunk(session, chunk, port))
                    tasks.append(task)
                    
                results = await asyncio.gather(*tasks)
                
            # Combine results
            all_embeddings = []
            for chunk_result in results:
                all_embeddings.extend(chunk_result['embeddings'])
                
            # Scale down if we scaled up
            if needed_workers > 1:
                await self.scale_workers(1)
                
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Error processing texts: {str(e)}")
            raise
            
    async def _process_chunk(
        self,
        session: aiohttp.ClientSession,
        texts: List[str],
        port: int
    ) -> Dict[str, Any]:
        """Process a chunk of texts using a specific worker."""
        try:
            async with session.post(
                f"http://localhost:{port}/embed384",
                json={"texts": texts, "batch_size": self.worker_configs[0].batch_size}
            ) as response:
                if response.status != 200:
                    raise Exception(f"Worker failed: {await response.text()}")
                return await response.json()
                
        except Exception as e:
            logger.error(f"Error processing chunk on port {port}: {str(e)}")
            raise 