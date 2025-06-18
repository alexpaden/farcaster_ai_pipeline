"""
Thread Classification Workflow with Auto-Scaling Ollama Support

This workflow leverages Ollama's OLLAMA_NUM_PARALLEL capability for high-throughput
classification. It can automatically manage Ollama instances based on workload.

Features:
- Auto-scales Ollama instances based on pending work
- Distributes workers across multiple Ollama instances
- Processes batches concurrently to maximize GPU/CPU utilization

Usage:
    # Manual mode - you manage Ollama
    python classifications_workflow.py --batch-size 8 --workers 4
    
    # Auto-scale mode - script manages Ollama instances
    python classifications_workflow.py --auto-scale --workers 8
       
Auto-scaling thresholds:
- >10,000 pending: 4 Ollama instances (max)
- >5,000 pending: 2 Ollama instances
- >0 pending: 1 Ollama instance
- 0 pending: Exits immediately

Key optimization: batch_size should match OLLAMA_NUM_PARALLEL for optimal throughput.
"""

import asyncio
import logging
from pathlib import Path
import os
import argparse
from typing import List, Tuple, Optional, Dict, Any
import json
import httpx
import time
import subprocess
import signal
import atexit

from src.db.connect import db

# threads_status values (continuing from embedding workflow):
# Embedding: 1=ready/retry, 2=processing, 3=done, 4=api_failed, 5=blank_text
# Classifications: Status 3 means ready for classification, then:
#   11=processing, 12=done, 10=retry

# Configure logging - default to INFO for progress updates
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Reduce httpx verbosity while keeping our progress logs
logging.getLogger('httpx').setLevel(logging.WARNING)

# Constants
DB_COMMAND_TIMEOUT = 300
MODEL = "gemma3:1b"  # Fixed model for classifications
DEFAULT_BATCH_SIZE = 8  # Match OLLAMA_NUM_PARALLEL for optimal throughput
DEFAULT_MAX_BATCHES = 0  # 0 means unlimited - process all available rows
PROGRESS_LOG_INTERVAL = 1000  # Log progress every 1k rows
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_TIMEOUT = 60.0  # Timeout for Ollama requests
OLLAMA_NUM_PARALLEL = 8  # Should match your OLLAMA_NUM_PARALLEL env var

# Classification prompt template
CLASSIFICATION_PROMPT = """Analyze this thread and provide THREE classifications.
(look PRIMARILY at FIRST POST, use replies (denoted by ↳) for further contextual understanding):

THREAD TYPE 
<THREAD TYPE >
- creation-showcase: Showing something they made (food, music, art, hobbies, artistic expression, personal vs professional, etc)
- project-update: Announcing updates/milestones for their project (companies, builds, releases)
- advice-giving: Sharing tips or recommendations
- question-seeking: Asking for help or information
- life-update: Sharing personal news
- opinion-stating: Expressing views or reactions
- ritual-greeting: GM/GN/social rituals
- call-to-action: Mint this/Vote/Retweet to win/Join raid
- market-alert: Price moves/exploits/time-sensitive alpha
- noise: Spam or unclear
</THREAD TYPE>

VALUE EXCHANGE
<VALUE EXCHANGE> (what participants PRIMARILY GAINED - look beyond surface information):
- knowledge-transfer: Learned something specific (pure information)
- social-capital: Built reputation, trust, or connections (information shared to establish credibility/relationships)
- emotional-energy: Got support/validation/vibes
- economic-value: Tokens/NFTs/airdrops/bounties
- none: No clear value exchanged
</VALUE EXCHANGE>

INTERACTION PATTERN
<INTERACTION PATTERN> (how people REPLIED - prioritize the dominant social dynamic):
- problem-solving: Working together on solutions
- social-bonding: Jokes/support/relationship building (includes playful exchanges, humor, personal references)
- show-and-tell: Presenting and getting reactions
- information-seeking: Q&A or learning (pure information exchange)
- debate: Arguments/disagreements/controversy
</INTERACTION PATTERN>

ANALYSIS GUIDANCE:
<ANALYSIS GUIDANCE>
- Look for the PRIMARY purpose, not just surface content
- Replies are ordered by popularity/engagement - use them to understand the thread's true impact
- In crypto/web3 contexts, information sharing often serves social capital building
- Personal references, humor, and self-deprecating responses signal social-bonding
- Consider what participants actually gained from the exchange beyond literal information
- When multiple patterns exist, choose the one that drives the most engagement/responses
</ANALYSIS GUIDANCE>

Thread:
<thread-blob>
{text}
</thread-blob>

###

Return EXACTLY:
thread_type:<one_of_above>
value_exchange:<one_of_above>
interaction_pattern:<one_of_above>"""

class OllamaManager:
    """Manages Ollama instances based on workload"""
    def __init__(self):
        self.instances = {}  # port -> process
        self.base_port = 11434
        self.max_instances = 4  # Adjust based on your Mac Studio specs
        
    async def get_pending_count(self) -> int:
        """Get count of threads pending classification"""
        try:
            async with db.pool.acquire() as conn:
                result = await conn.fetchval("""
                    SELECT count(*)
                    FROM   unbias.threads_subset
                    WHERE  spam = 2
                      AND  threads_status = 3
                """)
                return result
        except Exception as e:
            logger.error(f"Failed to get pending count: {e}")
            return 0
    
    def check_existing_instance(self, port: int) -> bool:
        """Check if Ollama is already running on port"""
        try:
            response = httpx.get(f'http://localhost:{port}/api/tags', timeout=2)
            if response.status_code == 200:
                # Also check if our model is available
                models = response.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                if not any(MODEL in name for name in model_names):
                    logger.warning(f"Model {MODEL} not found on port {port}. Available: {model_names}")
                    logger.info(f"Pull it with: ollama pull {MODEL}")
                return True
            return False
        except:
            return False
    
    def start_instance(self, port: int, num_parallel: int = 8) -> subprocess.Popen:
        """Start an Ollama instance on specified port"""
        # Check if already running
        if self.check_existing_instance(port):
            logger.info(f"Ollama already running on port {port}, using existing instance")
            return None  # Signal that we're using existing instance
        
        env = os.environ.copy()
        env['OLLAMA_HOST'] = f'127.0.0.1:{port}'
        env['OLLAMA_NUM_PARALLEL'] = str(num_parallel)
        
        logger.info(f"Starting Ollama on port {port} with NUM_PARALLEL={num_parallel}")
        process = subprocess.Popen(
            ['ollama', 'serve'],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        # Wait a bit for startup
        time.sleep(2)
        
        # Verify it's running
        if self.check_existing_instance(port):
            logger.info(f"Ollama started successfully on port {port}")
            return process
        
        # If not running, kill and raise
        process.kill()
        raise RuntimeError(f"Failed to start Ollama on port {port}")
    
    def stop_instance(self, port: int):
        """Stop an Ollama instance"""
        if port in self.instances:
            process = self.instances[port]
            if process is None:
                # This was an existing instance we didn't start
                logger.info(f"Skipping stop for existing Ollama on port {port}")
            else:
                logger.info(f"Stopping Ollama on port {port}")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
            del self.instances[port]
    
    def stop_all(self):
        """Stop all managed Ollama instances"""
        ports = list(self.instances.keys())
        for port in ports:
            self.stop_instance(port)
    
    async def adjust_instances(self, target_workers: int = None) -> List[int]:
        """Adjust number of Ollama instances based on workload"""
        if target_workers is None:
            # Auto-calculate based on pending work
            pending = await self.get_pending_count()
            logger.info(f"Pending classifications: {pending:,}")
            
            if pending > 10000:
                target_instances = min(self.max_instances, 4)
            elif pending > 5000:
                target_instances = 2
            elif pending > 0:
                target_instances = 1
            else:
                target_instances = 0
        else:
            # Use specified number of workers
            target_instances = min(target_workers, self.max_instances)
        
        current_instances = len(self.instances)
        
        # Scale up
        while len(self.instances) < target_instances:
            port = self.base_port + len(self.instances)
            try:
                process = self.start_instance(port)
                # Store process (None for existing instances we didn't start)
                self.instances[port] = process
            except Exception as e:
                logger.error(f"Failed to start instance on port {port}: {e}")
                break
        
        # Scale down
        while len(self.instances) > target_instances:
            port = max(self.instances.keys())
            self.stop_instance(port)
        
        return list(self.instances.keys())
    
    def get_urls(self) -> List[str]:
        """Get URLs of running instances"""
        return [f"http://localhost:{port}" for port in self.instances.keys()]


class ProgressTracker:
    """Shared progress tracker for all workers"""
    def __init__(self):
        self.total_rows = 0
        self.last_milestone = 0
        self.start_time = time.time()
        self.lock = asyncio.Lock()
        self.first_update = True
        
    async def add_rows(self, count: int):
        """Add processed rows and log if milestone reached"""
        async with self.lock:
            self.total_rows += count
            
            # Log first update to show system is working
            if self.first_update:
                self.first_update = False
                elapsed = time.time() - self.start_time
                logger.info(f"Processing started: {self.total_rows:,} rows completed in {elapsed:.1f}s")
            
            # Check if we've crossed a milestone
            current_milestone = (self.total_rows // PROGRESS_LOG_INTERVAL) * PROGRESS_LOG_INTERVAL
            if current_milestone > self.last_milestone:
                elapsed = time.time() - self.start_time
                rate = self.total_rows / elapsed if elapsed > 0 else 0
                # Use INFO level for progress updates
                logger.info(f"Progress: {self.total_rows:,} rows processed in {elapsed:.1f}s ({rate:.0f} rows/sec)")
                self.last_milestone = current_milestone

class ThreadClassificationWorker:
    """Worker class for processing thread classifications"""
    
    def __init__(self, worker_id: int, progress_tracker: ProgressTracker,
                 batch_size: int = DEFAULT_BATCH_SIZE, max_batches: int = 0, test_mode: bool = False,
                 ollama_url: str = OLLAMA_BASE_URL):
        self.worker_id = worker_id
        self.progress_tracker = progress_tracker
        self.batch_size = batch_size
        self.max_batches = max_batches
        self.test_mode = test_mode
        self.batches_processed = 0
        self.ollama_url = ollama_url
        self.logger = logging.LoggerAdapter(logger, {'worker': f'Worker-{worker_id}'})
        
        # Create HTTP client for Ollama with proper connection pooling
        self.client = httpx.AsyncClient(
            base_url=self.ollama_url,
            timeout=OLLAMA_TIMEOUT,
            limits=httpx.Limits(
                max_connections=OLLAMA_NUM_PARALLEL * 2,  # Allow some headroom
                max_keepalive_connections=OLLAMA_NUM_PARALLEL
            )
        )
        
    async def claim_batch(self) -> List[Tuple[bytes, str]]:
        """Claim a batch of threads for classification processing
        
        Returns:
            List of (hash, blob) tuples where hash is bytes and blob is str
        """
        try:
            async with db.pool.acquire() as conn:
                async with conn.transaction():
                    # Use test limit if in test mode
                    limit = min(5, self.batch_size) if self.test_mode else self.batch_size
                    
                    # Claim batch with FOR UPDATE SKIP LOCKED for parallel processing
                    # Only process threads that have completed embedding (status=3)
                    rows = await conn.fetch("""
                        WITH batch AS (
                            SELECT hash, blob
                            FROM   unbias.threads_subset
                            WHERE  threads_status = 3  -- Completed embeddings, ready for classification
                              AND  spam = 2
                            ORDER  BY timestamp DESC
                            LIMIT  $1
                            FOR UPDATE SKIP LOCKED
                        )
                        UPDATE unbias.threads_subset t
                        SET    threads_status = 11  -- Processing classifications
                        FROM   batch b
                        WHERE  t.hash = b.hash
                        RETURNING b.hash, b.blob;
                    """, limit)
                    
                    # Return with consistent types - hash is already memoryview/bytes from asyncpg
                    # Convert memoryview to bytes for consistency
                    return [(bytes(row['hash']), row['blob']) for row in rows]
        except Exception as e:
            self.logger.error(f"Database error claiming batch: {e}")
            raise
    
    async def classify_text(self, text: str, max_retries: int = 3) -> Optional[Dict[str, Any]]:
        """Classify a single text using Ollama with retries
        
        Returns:
            Dict with classification results or None if failed
        """
        if not text or not text.strip():
            return None
            
        # Clean text and prepare prompt
        cleaned_text = text.replace("↳", "-").strip()
        prompt = CLASSIFICATION_PROMPT.format(text=cleaned_text)
        
        # Valid values for each field
        thread_types = ['creation-showcase', 'project-update', 'advice-giving', 'question-seeking', 
                      'life-update', 'opinion-stating', 'ritual-greeting', 'call-to-action', 
                      'market-alert', 'noise']
        value_exchanges = ['knowledge-transfer', 'social-capital', 'emotional-energy', 
                         'economic-value', 'none']
        interaction_patterns = ['problem-solving', 'social-bonding', 'show-and-tell', 
                              'information-seeking', 'debate']
        
        # Try up to max_retries times
        for retry in range(max_retries):
            try:
                response = await self.client.post("/api/generate", json={
                    "model": MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        # Generation parameters for consistent classification
                        "temperature": 0.1 if retry == 0 else 0.3,  # Increase temp on retries
                        "top_p": 0.9,
                        "top_k": 40,
                        "num_predict": 50,  # Just need 3 lines
                        "seed": 42 + retry,  # Different seed each retry
                        
                        # Performance parameters
                        "num_ctx": 8192,  # Increased for long prompt
                        "num_gpu": 999,   # Use all GPU layers
                    }
                })
                
                if response.status_code != 200:
                    continue
                
                result = response.json()
                response_text = result.get("response", "").strip()
                
                # Parse simple response format
                classifications = {}
                
                # Look for each classification in the response (case insensitive)
                import re
                lines = response_text.lower().split('\n')
                
                for line in lines:
                    # Thread type
                    if 'thread_type' in line and ':' in line:
                        value = line.split(':', 1)[1].strip()
                        if value in thread_types:
                            classifications['thread_type'] = value
                    
                    # Value exchange
                    elif 'value_exchange' in line and ':' in line:
                        value = line.split(':', 1)[1].strip()
                        if value in value_exchanges:
                            classifications['value_exchange'] = value
                    
                    # Interaction pattern
                    elif 'interaction_pattern' in line and ':' in line:
                        value = line.split(':', 1)[1].strip()
                        if value in interaction_patterns:
                            classifications['interaction_pattern'] = value
                
                # Check if we got all fields
                required_fields = ["thread_type", "value_exchange", "interaction_pattern"]
                if all(field in classifications for field in required_fields):
                    return classifications
                else:
                    missing = [f for f in required_fields if f not in classifications]
                    if retry < max_retries - 1:
                        self.logger.debug(f"Retry {retry + 1}: Missing {missing}. Response: {response_text}")
                    else:
                        self.logger.warning(f"Final attempt: Missing {missing}. Response: {response_text}")
                        
            except Exception as e:
                if retry < max_retries - 1:
                    self.logger.debug(f"Retry {retry + 1}: Ollama request failed: {e}")
                else:
                    self.logger.warning(f"Final attempt: Ollama request failed: {e}")
        
        return None
    
    async def process_batch(self, batch: List[Tuple[bytes, str]]) -> Tuple[List[Tuple[bytes, Dict[str, Any]]], List[bytes]]:
        """Process a batch of texts concurrently using Ollama's parallel processing"""
        if not batch:
            return [], []
        
        # Separate blank texts
        to_process = []
        blank_hashes = []
        
        for hash_val, text in batch:
            if text and text.strip():
                to_process.append((hash_val, text))
            else:
                blank_hashes.append(hash_val)
        
        if not to_process:
            return [], blank_hashes
        
        # Process all texts concurrently
        tasks = []
        for hash_val, text in to_process:
            tasks.append(self.classify_text(text))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Pair results with hashes
        successful = []
        failed = []
        
        for i, result in enumerate(results):
            hash_val = to_process[i][0]
            if isinstance(result, Exception):
                self.logger.warning(f"Classification failed: {result}")
                failed.append(hash_val)
            elif isinstance(result, dict):
                successful.append((hash_val, result))
            else:
                failed.append(hash_val)
        
        return successful, blank_hashes + failed
    
    async def update_classifications(self, successful: List[Tuple[bytes, Dict[str, Any]]]):
        """Update database with successful classifications
        
        Args:
            successful: List of (hash, classifications) tuples where hash is bytes
        """
        if not successful:
            return
        
        try:
            async with db.pool.acquire() as conn:
                async with conn.transaction():
                    # Prepare data for bulk update
                    update_data = []
                    for hash_val, classifications in successful:
                        update_data.append((
                            classifications.get("thread_type"),
                            classifications.get("value_exchange"),
                            classifications.get("interaction_pattern"),
                            hash_val
                        ))
                    
                    # Bulk update using executemany
                    await conn.executemany("""
                        UPDATE unbias.threads_subset
                        SET thread_type = $1,
                            value_exchange = $2,
                            interaction_pattern = $3,
                            threads_status = 12,  -- Done
                            classification_process_timestamp = NOW()
                        WHERE hash = $4;
                    """, update_data)
                    
                    # Update progress tracker
                    await self.progress_tracker.add_rows(len(successful))
                    
        except Exception as e:
            self.logger.error(f"Database error updating classifications: {e}")
            raise
    
    async def mark_failed(self, failed_hashes: List[bytes]):
        """Mark failed items for retry"""
        if not failed_hashes:
            return
        try:
            async with db.pool.acquire() as conn:
                await conn.execute("""
                    UPDATE unbias.threads_subset
                    SET    threads_status = 10
                    WHERE  hash = ANY($1);
                """, failed_hashes)
        except Exception as e:
            self.logger.error(f"Database error marking failures: {e}")
    
    async def run(self):
        """Main worker loop"""
        current_batch_hashes = []
        total_claim_time = 0
        total_api_time = 0
        total_update_time = 0
        total_batches = 0
        
        try:
            while self.max_batches == 0 or self.batches_processed < self.max_batches:
                claim_start = time.time()
                batch = await self.claim_batch()
                claim_time = time.time() - claim_start
                total_claim_time += claim_time
                
                if not batch:
                    self.logger.info(f"Worker {self.worker_id} ({self.ollama_url}) - No more batches available")
                    break
                
                current_batch_hashes = [item[0] for item in batch]
                
                api_start = time.time()
                successful, failed = await self.process_batch(batch)
                api_time = time.time() - api_start
                total_api_time += api_time
                
                update_start = time.time()
                await self.update_classifications(successful)
                if failed:
                    await self.mark_failed(failed)
                update_time = time.time() - update_start
                total_update_time += update_time
                
                current_batch_hashes = []
                self.batches_processed += 1
                total_batches += 1
                
                # Log performance stats
                if api_time > 0:
                    items_per_second = len(batch) / api_time
                    self.logger.info(
                        f"Worker {self.worker_id} batch {self.batches_processed}: "
                        f"{len(successful)}/{len(batch)} success, "
                        f"api={api_time:.2f}s ({items_per_second:.1f} items/sec)"
                    )
                
                if self.test_mode:
                    break
                    
        except Exception as e:
            self.logger.error(f"Worker {self.worker_id} crashed: {e}", exc_info=True)
            raise
        finally:
            # Close HTTP client
            await self.client.aclose()
            
            # Mark any in-progress items for retry
            if current_batch_hashes:
                self.logger.warning(f"Worker {self.worker_id} crashed with {len(current_batch_hashes)} items in progress, marking for retry")
                try:
                    await self.mark_failed(current_batch_hashes)
                except Exception as e:
                    self.logger.error(f"Failed to mark crashed batch for retry: {e}")


async def main(args):
    """Main entry point for the threads classification pipeline"""
    # Initial startup message at INFO level
    logger.info("Starting threads classification pipeline...")
    start_time = time.time()
    
    # Create Ollama manager
    ollama_manager = OllamaManager()
    
    # Register cleanup on exit
    atexit.register(ollama_manager.stop_all)
    signal.signal(signal.SIGINT, lambda sig, frame: (ollama_manager.stop_all(), exit(0)))
    signal.signal(signal.SIGTERM, lambda sig, frame: (ollama_manager.stop_all(), exit(0)))
    
    try:
        # Initialize the database connection pool
        await db.initialize_pool(command_timeout=DB_COMMAND_TIMEOUT)
        
        # Start Ollama instances based on workload or args
        if args.auto_scale:
            ports = await ollama_manager.adjust_instances()
            if not ports:
                logger.info("No pending classifications found")
                return
        else:
            # Manual mode - start instances for workers
            num_instances = max(1, args.workers // 2)  # 2 workers per instance
            ports = await ollama_manager.adjust_instances(target_workers=num_instances)
        
        logger.info(f"Running with {len(ports)} Ollama instance(s) on ports: {ports}")
        
        # Create progress tracker
        progress_tracker = ProgressTracker()
        
        # Create workers distributed across Ollama instances
        workers = []
        urls = ollama_manager.get_urls()
        
        for i in range(args.workers):
            # Distribute workers across available Ollama instances
            ollama_url = urls[i % len(urls)]
            
            worker = ThreadClassificationWorker(
                worker_id=i,
                progress_tracker=progress_tracker,
                batch_size=args.batch_size,
                max_batches=args.max_batches,
                test_mode=args.test,
                ollama_url=ollama_url
            )
            workers.append(worker)
            logger.info(f"Worker {i} using {ollama_url}")
        
        # Log configuration
        logger.info(f"Starting {args.workers} workers with batch_size={args.batch_size}, model={MODEL}")
        if args.test:
            logger.info("Running in TEST MODE - will process maximum 5 rows per worker")
        
        # Run workers in parallel
        await asyncio.gather(*[worker.run() for worker in workers])
        
        # Final summary
        total_duration = time.time() - start_time
        total_rows = progress_tracker.total_rows
        rate = total_rows / total_duration if total_duration > 0 else 0
        logger.info(f"Classifications pipeline completed: {total_rows:,} rows in {total_duration:.1f}s ({rate:.0f} rows/sec)")
        
    except Exception as e:
        logger.error(f"An error occurred in the main workflow: {e}", exc_info=True)
    finally:
        # Stop all Ollama instances
        ollama_manager.stop_all()
        
        # Ensure the database pool is closed
        if db._pool:
            await db.close_pool()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Thread Classification Batch Processor',
        epilog='Auto-scales Ollama instances based on workload'
    )
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
                       help=f'Concurrent requests per batch - should match OLLAMA_NUM_PARALLEL (default: {DEFAULT_BATCH_SIZE})')
    parser.add_argument('--max-batches', type=int, default=DEFAULT_MAX_BATCHES,
                       help='Maximum number of batches per worker (default: 0, unlimited)')
    parser.add_argument('--workers', type=int, default=2,
                       help='Number of workers claiming batches (default: 2)')
    parser.add_argument('--test', action='store_true',
                       help='Run in test mode (process max 5 rows per worker)')
    parser.add_argument('--auto-scale', action='store_true',
                       help='Automatically scale Ollama instances based on workload (>10k rows = max instances)')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
