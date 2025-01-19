"""
Centralized logging for the embedding pipeline.
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional, Deque, Dict
from collections import deque
from .utils import format_number, get_memory_usage
import asyncio

@dataclass
class MetricsBase:
    """Base class for all metrics."""
    start_time: float = field(default_factory=time.time)
    total_time: float = 0.0

@dataclass
class InferenceMetrics(MetricsBase):
    """Track detailed inference metrics."""
    tokenization_time: float = 0.0
    model_forward_time: float = 0.0
    pooling_time: float = 0.0
    quantization_time: float = 0.0
    batch_size: int = 0
    texts_per_second: float = 0.0

@dataclass
class BatchProgress:
    """Track progress within a batch."""
    batch_size: int
    processed: int = 0
    start_time: float = field(default_factory=time.time)
    last_log_time: float = field(default_factory=time.time)
    inference_times: List[float] = field(default_factory=list)
    db_times: List[float] = field(default_factory=list)

class PipelineLogger:
    """Consolidated logging for the embedding pipeline."""
    def __init__(self):
        self.start_time = time.time()
        self.last_log_time = self.start_time
        self.last_memory_log = self.start_time
        self.total_processed = 0
        self.total_remaining = 0
        
        # DB update tracking
        self.pending_updates = 0
        self.completed_updates = 0
        self.db_chunk_times = []
        self.last_db_log = time.time()
        
        # Rolling window stats
        self.window_start = time.time()
        self.window_processed = 0
        self.window_updates = 0
        self.last_60s_measurements = []
        
        # Rate tracking (last 60 seconds)
        self.inference_updates = deque(maxlen=60)  # (timestamp, count) tuples
        self.db_updates = deque(maxlen=60)  # (timestamp, count) tuples
        
        # Batch monitoring
        self.last_batch_stats: Dict = {}
        self.last_batch_check = time.time()
        self.last_update_duration = 0.0
        self.last_processed_id = 0
        self.db_queue = None  # Set by process_casts
        
    def get_rate(self, updates: Deque) -> float:
        """Calculate rate from update deque."""
        now = time.time()
        # Remove old entries
        while updates and updates[0][0] < now - 60:
            updates.popleft()
        
        if len(updates) < 2:
            return 0.0
            
        first_time, first_count = updates[0]
        last_time, last_count = updates[-1]
        time_diff = last_time - first_time
        if time_diff <= 0:
            return 0.0
            
        return (last_count - first_count) / time_diff
        
    def get_inference_rate(self) -> float:
        """Get current inference rate in rows/sec."""
        return self.get_rate(self.inference_updates)
        
    def get_db_rate(self) -> float:
        """Get current DB update rate in rows/sec."""
        return self.get_rate(self.db_updates)
    
    def log_memory(self, force: bool = False):
        """Log memory usage if enough time has passed."""
        now = time.time()
        if force or now - self.last_memory_log >= 30:
            process_mem, gpu_mem = get_memory_usage()
            print(f"Memory: Process={format_number(process_mem)}MB, GPU={format_number(gpu_mem)}MB")
            self.last_memory_log = now
            
    def log_status_update(self, force: bool = False):
        """Log comprehensive status update."""
        now = time.time()
        if not force and now - self.last_log_time < 5:
            return
            
        elapsed = now - self.start_time
        
        # Calculate throughput metrics
        window_time = now - self.window_start
        if window_time >= 5:
            window_tps = self.window_processed / window_time if window_time > 0 else 0
            window_db_tps = self.window_updates / window_time if window_time > 0 else 0
            
            # Reset window
            self.window_start = now
            self.window_processed = 0
            self.window_updates = 0
        else:
            window_tps = 0
            window_db_tps = 0
            
        # Calculate rolling 60s TPS
        self.last_60s_measurements = [(t, c) for t, c in self.last_60s_measurements if t > now - 60]
        rolling_60s_tps = 0
        if len(self.last_60s_measurements) >= 2:
            first_time, first_count = self.last_60s_measurements[0]
            last_time, last_count = self.last_60s_measurements[-1]
            time_diff = last_time - first_time
            if time_diff > 0:
                rolling_60s_tps = (last_count - first_count) / time_diff
                
        # Calculate overall metrics
        overall_tps = self.total_processed / elapsed if elapsed > 0 else 0
        remaining = max(0, self.total_remaining - self.total_processed)
        eta_minutes = remaining / rolling_60s_tps / 60 if rolling_60s_tps > 0 else 0
        
        # Calculate DB metrics
        recent_db_times = self.db_chunk_times[-10:] if self.db_chunk_times else []
        avg_chunk_time = sum(recent_db_times) / len(recent_db_times) if recent_db_times else 0
        
        # Get current rates
        inference_rate = self.get_inference_rate()
        db_rate = self.get_db_rate()
        
        # Log comprehensive status
        print(f"\n[Status Update | {time.strftime('%H:%M:%S')} | +{elapsed:.1f}s]")
        print("Pipeline Status:")
        print(f"  Inference Queue: {format_number(self.pending_updates - self.completed_updates)} updates pending")
        print(f"  DB Performance: {format_number(avg_chunk_time, 3)}s/chunk")
        print(f"  Inference Rate: {format_number(inference_rate)} rows/sec")
        print(f"  DB Update Rate: {format_number(db_rate)} rows/sec")
        print("\nProgress:")
        print(f"  Processed: {format_number(self.total_processed)} / {format_number(self.total_remaining)} rows")
        print(f"  Completion: {format_number(self.total_processed/self.total_remaining*100 if self.total_remaining else 0)}%")
        print(f"  Overall TPS: {format_number(overall_tps)}")
        print(f"  ETA: {format_number(eta_minutes)}m")
        
        self.last_log_time = now
        
    def log_db_chunk(self, duration: float, updated: int, total: int):
        """Log database chunk update with timing."""
        self.last_update_duration = duration
        now = time.time()
        self.db_chunk_times.append(duration)
        self.completed_updates += updated
        self.window_updates += updated
        self.db_updates.append((now, self.completed_updates))
        
        print(f"\n[DB Update] Chunk completed:")
        print(f"  Updated: {format_number(updated)}/{format_number(total)} rows")
        print(f"  Duration: {format_number(duration, 3)}s")
        print(f"  Speed: {format_number(updated/duration if duration > 0 else 0)} rows/sec")
        
        # Log overall status if enough time has passed
        if now - self.last_db_log >= 5:
            pending = self.pending_updates - self.completed_updates
            print(f"\n[DB Status] Updates pending: {format_number(pending)}")
            print(f"Total completed: {format_number(self.completed_updates)}")
            self.last_db_log = now
            
    def update_processed(self, count: int):
        """Update processed count and log progress."""
        now = time.time()
        self.total_processed += count
        self.window_processed += count
        self.inference_updates.append((now, self.total_processed))
        
        # Update rolling window
        self.last_60s_measurements.append((now, self.total_processed))
        
        # Log status update
        self.log_status_update()
        
    def queue_updates(self, count: int):
        """Track updates being queued for processing."""
        self.pending_updates += count
        
    def set_total_remaining(self, total: int):
        """Set total rows to be processed."""
        self.total_remaining = total
        print(f"\nInitializing pipeline for {format_number(total)} rows") 

    async def update_batch_stats(self, pool) -> None:
        """Update batch-level statistics."""
        try:
            async with pool.acquire() as conn:
                stats = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total_rows,
                        COUNT(*) FILTER (WHERE embedding384 IS NULL) as pending_updates,
                        COUNT(*) FILTER (WHERE embedding384_updated_at > now() - interval '5 minutes') as recent_updates,
                        MAX(embedding384_updated_at) as last_update_time
                    FROM casts
                    WHERE id > $1
                """, self.last_processed_id)
                
                self.last_batch_stats = dict(stats)
                
                # Check for slow updates
                if self.last_update_duration > 1.0:
                    txn_info = await conn.fetch("""
                        SELECT pid, 
                               now() - xact_start as duration,
                               wait_event_type,
                               wait_event,
                               query
                        FROM pg_stat_activity 
                        WHERE state != 'idle'
                          AND query LIKE '%embedding384%'
                    """)
                    if txn_info:
                        print(f"\n[Slow Update Investigation]"
                              f"\nActive Transactions: {len(txn_info)}"
                              f"\nDetails: {txn_info}")
                
        except Exception as e:
            print(f"Error updating batch stats: {e}")
    
    def log_batch_summary(self) -> None:
        """Log batch-level summary."""
        now = time.time()
        if now - self.last_batch_check < 60:  # Only log every 1 minute
            return
            
        if not self.last_batch_stats:
            return
            
        queue_size = self.db_queue.qsize() if self.db_queue else 0
        queue_capacity = self.db_queue.maxsize if self.db_queue else 0
        
        print(f"\n[Batch Summary]"
              f"\nPending Updates: {format_number(self.last_batch_stats['pending_updates'])}"
              f"\nRecent Updates (5m): {format_number(self.last_batch_stats['recent_updates'])}"
              f"\nAvg Update Speed: {format_number(self.get_db_rate())} rows/sec"
              f"\nQueue Status: {queue_size}/{queue_capacity}")
        
        self.last_batch_check = now
    
    def set_db_queue(self, queue: asyncio.Queue) -> None:
        """Set the DB queue reference for monitoring."""
        self.db_queue = queue
    
    def update_processed(self, count: int):
        """Update processed count and log progress."""
        now = time.time()
        self.total_processed += count
        self.window_processed += count
        self.inference_updates.append((now, self.total_processed))
        
        # Update rolling window
        self.last_60s_measurements.append((now, self.total_processed))
        
        # Log status update
        self.log_status_update()
        
    def queue_updates(self, count: int):
        """Track updates being queued for processing."""
        self.pending_updates += count
        
    def set_total_remaining(self, total: int):
        """Set total rows to be processed."""
        self.total_remaining = total
        print(f"\nInitializing pipeline for {format_number(total)} rows") 