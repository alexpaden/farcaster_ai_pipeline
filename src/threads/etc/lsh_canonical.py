"""
LSH-based thread deduplication system.
Finds similar threads and assigns canonical_hash pointing to first similar thread.
"""

import asyncio
import hashlib
import re
import time
from collections import defaultdict
from typing import Set, List, Dict, Optional, Tuple
import logging

# Import database connection
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db.connect import db

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MinHashLSH:
    """MinHash-based Locality Sensitive Hashing for text similarity."""
    
    def __init__(self, num_perm: int = 128, threshold: float = 0.9):
        """
        Initialize LSH with MinHash.
        
        Args:
            num_perm: Number of permutations for MinHash (higher = more accurate)
            threshold: Similarity threshold (0.9 = 90% similar)
        """
        self.num_perm = num_perm
        self.threshold = threshold
        self.bands = self._get_num_bands()
        self.rows_per_band = self.num_perm // self.bands
        self.hash_tables = [defaultdict(list) for _ in range(self.bands)]
        self.signatures = {}  # hash -> minhash signature
        self.thread_hashes = {}  # normalized_text -> original_hash
        
    def _get_num_bands(self) -> int:
        """Calculate number of bands for LSH based on threshold."""
        # For threshold t, we want (1/b)^r ≈ t where b*r = num_perm
        # This gives us optimal band/row configuration
        if self.threshold >= 0.9:
            return 16  # 16 bands × 8 rows = 128 permutations
        elif self.threshold >= 0.8:
            return 21  # 21 bands × 6 rows ≈ 128 permutations  
        else:
            return 32  # 32 bands × 4 rows = 128 permutations
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for consistent comparison."""
        if not text:
            return ""

        text = text.lower()

        # 1. Strip leading "@user: " prefix
        text = re.sub(r'^@[a-z0-9_.-]+:\s*', '', text)

        # 2. Remove all @mentions anywhere
        text = re.sub(r'@[a-z0-9_.-]+\b', '', text)

        # 3. Kill urls (optional but usually noise)
        text = re.sub(r'https?://\S+', '', text)

        # 4. Collapse whitespace first so punctuation regex is simpler
        text = re.sub(r'\s+', ' ', text).strip()

        # 5. Remove punctuation (keep hashtags if you care)
        text = re.sub(r'[^\w\s#]', '', text)

        return text
    
    def _get_shingles(self, text: str, k: int = 3) -> Set[str]:
        """Generate k-shingles (k-grams) from text."""
        normalized = self._normalize_text(text)
        if len(normalized) < k:
            # For short text, also include individual words as "shingles"
            words = normalized.split()
            return set(words + [normalized])  # Both words and full text
        
        shingles = set()
        for i in range(len(normalized) - k + 1):
            shingles.add(normalized[i:i+k])
        
        return shingles
    
    def _minhash_signature(self, shingles: Set[str]) -> List[int]:
        """Generate MinHash signature for a set of shingles."""
        if not shingles:
            return [0] * self.num_perm
        
        signature = []
        for i in range(self.num_perm):
            min_hash = float('inf')
            for shingle in shingles:
                # Use different hash seeds for each permutation
                hash_val = hash(f"{i}_{shingle}") % (2**32)
                min_hash = min(min_hash, hash_val)
            signature.append(min_hash)
        
        return signature
    
    def _jaccard_similarity(self, sig1: List[int], sig2: List[int]) -> float:
        """Estimate Jaccard similarity from MinHash signatures."""
        if len(sig1) != len(sig2):
            return 0.0
        
        matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
        return matches / len(sig1)
    
    def add_thread(self, thread_hash: str, text: str) -> Optional[str]:
        """
        Add thread to LSH index and check for similar threads.
        
        Args:
            thread_hash: Unique hash of the thread
            text: Thread content
            
        Returns:
            Hash of similar thread if found, None if this is unique
        """
        norm = self._normalize_text(text)
        thresh = 0.95 if len(norm) > 60 else 0.90     # Dynamic threshold based on length
        
        # Quick check for exact normalized matches first
        if norm in self.thread_hashes:
            return self.thread_hashes[norm]
        
        # Generate shingles and MinHash signature
        shingles = self._get_shingles(text)
        signature = self._minhash_signature(shingles)
        
        # ---- banded lookup -----------------------------------------------------
        candidate_hashes = set()
        for i, table in enumerate(self.hash_tables):
            start = i * self.rows_per_band
            band = tuple(signature[start:start + self.rows_per_band])
            candidate_hashes.update(table.get(band, []))
        # ------------------------------------------------------------------------
        
        # Compare only against candidates
        for cand in candidate_hashes:
            if self._jaccard_similarity(signature, self.signatures[cand]) >= thresh:
                logger.debug(f"Found similar thread: {thread_hash} -> {cand} (similarity: {self._jaccard_similarity(signature, self.signatures[cand]):.3f}, thresh: {thresh})")
                return cand
        
        # Unique → index it
        self.signatures[thread_hash] = signature
        self.thread_hashes[norm] = thread_hash
        for i, table in enumerate(self.hash_tables):
            start = i * self.rows_per_band
            band = tuple(signature[start:start + self.rows_per_band])
            table[band].append(thread_hash)
        
        return None  # This thread is canonical (unique)


class ThreadDeduplicator:
    """Main class for thread deduplication using LSH."""
    
    def __init__(self, similarity_threshold: float = 0.9, batch_size: int = 100000):
        """
        Initialize thread deduplicator.
        
        Args:
            similarity_threshold: Minimum similarity to consider threads duplicates
            batch_size: Number of threads to process in each batch
        """
        self.lsh = MinHashLSH(threshold=similarity_threshold)
        self.batch_size = batch_size
        self.stats = {
            'total_processed': 0,
            'duplicates_found': 0,
            'canonicals_created': 0,
            'batches_processed': 0,
            'db_fetch_time': 0.0,
            'lsh_processing_time': 0.0,
            'db_update_time': 0.0
        }
    
    async def process_threads_subset(self):
        """Process all threads in the subset table and assign canonical_hash."""
        logger.info("Starting thread deduplication process...")
        start_time = time.time()
        
        # Initialize database pool
        await db.initialize_pool()
        
        try:
            # Get total count for progress tracking
            async with db.pool.acquire() as conn:
                total_count = await conn.fetchval(
                    "SELECT COUNT(*) FROM unbias.threads_subset WHERE threads_status = 3 AND spam = 2"
                )
                logger.info(f"Processing {total_count:,} threads...")
            
            # Process threads in timestamp order (oldest first)
            offset = 0
            while True:
                batch_processed = await self._process_batch(offset)
                if batch_processed == 0:
                    break
                
                offset += self.batch_size
                self.stats['batches_processed'] += 1
            
            # Final statistics
            elapsed = time.time() - start_time
            await self._log_final_stats(elapsed)
            
        finally:
            await db.close_pool()
    
    async def _process_batch(self, offset: int) -> int:
        """Process a batch of threads."""
        batch_start_time = time.time()
        
        # Database fetch timing
        fetch_start = time.time()
        async with db.pool.acquire() as conn:
            # Fetch batch ordered by timestamp (oldest first)
            # Use indexes: threads_status = 3, spam = 2
            rows = await conn.fetch("""
                SELECT hash, blob, timestamp
                FROM unbias.threads_subset
                WHERE canonical_hash IS NULL
                AND threads_status = 3
                AND spam = 2
                ORDER BY timestamp ASC
                LIMIT $1 OFFSET $2
            """, self.batch_size, offset)
        
        fetch_time = time.time() - fetch_start
        self.stats['db_fetch_time'] += fetch_time
        
        if not rows:
            return 0
        
        logger.info(f"Fetched {len(rows):,} rows in {fetch_time:.2f}s")
        
        # LSH processing timing
        lsh_start = time.time()
        updates = []
        for i, row in enumerate(rows):
            thread_hash_bytes = row['hash']  # This is bytes from database
            thread_hash = thread_hash_bytes.hex() if isinstance(thread_hash_bytes, bytes) else thread_hash_bytes  # Convert to hex string
            blob = row['blob'] or ""
            
            # Check for similar thread (LSH works with hex strings)
            similar_hash = self.lsh.add_thread(thread_hash, blob)
            
            if similar_hash:
                # This is a duplicate - point to canonical (convert back to bytes)
                canonical_hash_bytes = bytes.fromhex(similar_hash) if isinstance(similar_hash, str) else similar_hash
                self.stats['duplicates_found'] += 1
            else:
                # This is unique - it becomes its own canonical (convert back to bytes)
                canonical_hash_bytes = thread_hash_bytes
                self.stats['canonicals_created'] += 1
            
            updates.append((canonical_hash_bytes, thread_hash_bytes))
            self.stats['total_processed'] += 1
            
            # Progress every 10k threads
            if self.stats['total_processed'] % 10000 == 0:
                await self._log_frequent_progress()
        
        lsh_time = time.time() - lsh_start
        self.stats['lsh_processing_time'] += lsh_time
        
        logger.info(f"LSH processed {len(rows):,} threads in {lsh_time:.2f}s ({len(rows)/lsh_time:.0f} threads/sec)")
        
        # Database update timing
        update_start = time.time()
        if updates:
            async with db.pool.acquire() as conn:
                await conn.executemany("""
                    UPDATE unbias.threads_subset 
                    SET canonical_hash = $1 
                    WHERE hash = $2
                """, updates)
        
        update_time = time.time() - update_start
        self.stats['db_update_time'] += update_time
        
        batch_total_time = time.time() - batch_start_time
        logger.info(f"Updated {len(updates):,} rows in {update_time:.2f}s | Batch total: {batch_total_time:.2f}s")
        
        return len(rows)
    
    async def _log_frequent_progress(self):
        """Log progress every 10k threads."""
        processed = self.stats['total_processed']
        duplicates = self.stats['duplicates_found']
        canonicals = self.stats['canonicals_created']
        
        duplication_rate = (duplicates / processed * 100) if processed > 0 else 0
        
        # Calculate current LSH index size for memory monitoring
        lsh_signatures = len(self.lsh.signatures)
        lsh_normalized = len(self.lsh.thread_hashes)
        
        logger.info(f"📊 {processed:,} processed | {duplicates:,} dupes ({duplication_rate:.1f}%) | {canonicals:,} canonicals | LSH index: {lsh_signatures:,} sigs, {lsh_normalized:,} norms")

    async def _log_final_stats(self, elapsed_seconds: float):
        """Log final processing statistics."""
        total = self.stats['total_processed']
        duplicates = self.stats['duplicates_found']
        canonicals = self.stats['canonicals_created']
        
        duplication_rate = (duplicates / total * 100) if total > 0 else 0
        threads_per_second = total / elapsed_seconds if elapsed_seconds > 0 else 0
        
        # Detailed timing breakdown
        db_fetch_time = self.stats['db_fetch_time']
        lsh_time = self.stats['lsh_processing_time']
        db_update_time = self.stats['db_update_time']
        
        logger.info("=" * 80)
        logger.info("DEDUPLICATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total threads processed: {total:,}")
        logger.info(f"Duplicates found: {duplicates:,} ({duplication_rate:.1f}%)")
        logger.info(f"Canonical threads: {canonicals:,}")
        logger.info(f"Processing time: {elapsed_seconds:.1f}s")
        logger.info(f"Processing speed: {threads_per_second:.0f} threads/second")
        logger.info("")
        logger.info("⏱️  TIMING BREAKDOWN:")
        logger.info(f"   Database fetch: {db_fetch_time:.1f}s ({db_fetch_time/elapsed_seconds*100:.1f}%)")
        logger.info(f"   LSH processing: {lsh_time:.1f}s ({lsh_time/elapsed_seconds*100:.1f}%)")
        logger.info(f"   Database update: {db_update_time:.1f}s ({db_update_time/elapsed_seconds*100:.1f}%)")
        logger.info(f"   Other overhead: {elapsed_seconds-db_fetch_time-lsh_time-db_update_time:.1f}s")
        logger.info("")
        logger.info(f"🧠 LSH INDEX SIZE:")
        logger.info(f"   Signatures stored: {len(self.lsh.signatures):,}")
        logger.info(f"   Normalized texts: {len(self.lsh.thread_hashes):,}")
        logger.info(f"   Hash tables: {len(self.lsh.hash_tables)} bands")
        logger.info("=" * 80)
        
        # Provide recommendations based on duplication rate
        if duplication_rate >= 30:
            logger.info("✅ High duplication rate! LSH deduplication is very beneficial.")
        elif duplication_rate >= 20:
            logger.info("✅ Good duplication rate! LSH deduplication is beneficial.")
        elif duplication_rate >= 10:
            logger.info("⚠️  Moderate duplication rate. LSH provides some benefit.")
        else:
            logger.info("❌ Low duplication rate. Consider skipping deduplication.")


async def run_deduplication(similarity_threshold: float = 0.9):
    """
    Run thread deduplication on the threads_subset table.
    
    Args:
        similarity_threshold: Similarity threshold (0.9 = 90% similar)
    """
    deduplicator = ThreadDeduplicator(similarity_threshold=similarity_threshold)
    await deduplicator.process_threads_subset()


async def analyze_canonical_results():
    """Analyze the results of canonicalization."""
    logger.info("Analyzing canonicalization results...")
    
    await db.initialize_pool()
    
    try:
        async with db.pool.acquire() as conn:
            # Basic stats
            stats = await conn.fetchrow("""
                SELECT 
                    COUNT(*) as total_threads,
                    COUNT(DISTINCT canonical_hash) as unique_canonicals,
                    COUNT(*) - COUNT(DISTINCT canonical_hash) as duplicates
                FROM unbias.threads_subset
                WHERE canonical_hash IS NOT NULL
            """)
            
            # Top duplicated threads
            top_duplicates = await conn.fetch("""
                SELECT 
                    canonical_hash,
                    COUNT(*) as duplicate_count,
                    MIN(timestamp) as first_seen,
                    MAX(timestamp) as last_seen
                FROM unbias.threads_subset
                WHERE canonical_hash IS NOT NULL
                GROUP BY canonical_hash
                HAVING COUNT(*) > 1
                ORDER BY COUNT(*) DESC
                LIMIT 10
            """)
            
            logger.info("=" * 60)
            logger.info("CANONICALIZATION ANALYSIS")
            logger.info("=" * 60)
            logger.info(f"Total threads: {stats['total_threads']:,}")
            logger.info(f"Unique canonicals: {stats['unique_canonicals']:,}")
            logger.info(f"Duplicates: {stats['duplicates']:,}")
            
            if stats['total_threads'] > 0:
                dedup_rate = (stats['duplicates'] / stats['total_threads']) * 100
                logger.info(f"Deduplication rate: {dedup_rate:.1f}%")
            
            logger.info("\nTop 10 most duplicated threads:")
            for i, row in enumerate(top_duplicates, 1):
                logger.info(f"{i:2d}. {row['duplicate_count']:3d} copies | Hash: {row['canonical_hash'][:12]}...")
    
    finally:
        await db.close_pool()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Thread deduplication using LSH")
    parser.add_argument("--threshold", type=float, default=0.9, 
                       help="Similarity threshold (default: 0.9)")
    parser.add_argument("--analyze", action="store_true",
                       help="Analyze existing canonicalization results")
    
    args = parser.parse_args()
    
    if args.analyze:
        asyncio.run(analyze_canonical_results())
    else:
        asyncio.run(run_deduplication(args.threshold))
