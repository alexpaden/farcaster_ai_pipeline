#!/usr/bin/env python3
"""
Export threads data from PostgreSQL to sharded Parquet files for Hugging Face.
Uses the existing database connection from src.db.connect.
"""

import asyncio
import math
import os
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import asyncpg
from datetime import datetime
import logging

# Add parent directory to path to import from src
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.db.connect import db
from src.db.logger import setup_logging

# Set up logging
logger = setup_logging()

# Configuration
ROWS_PER_CHUNK = 5_000_000      # Fetch and write 5M rows at a time
OUTPUT_DIR = Path("../../data")   # Output directory

# The base query to execute (no ORDER BY for speed, but can add if needed)
BASE_QUERY = """
SELECT 
    hash, 
    fids, 
    reactions, 
    author_fid, 
    "timestamp", 
    claimed_at, 
    "blob", 
    blob_embedding, 
    blob_embedding_fp16
FROM unbias.threads
WHERE threads_status = 3 AND spam = 2
ORDER BY timestamp
LIMIT {limit} OFFSET {offset}
"""


async def get_total_row_count(conn):
    """Get the total number of rows that will be exported."""
    count_query = """
    SELECT COUNT(*)
    FROM unbias.threads
    WHERE threads_status = 3 AND spam = 2
    """
    result = await conn.fetchval(count_query)
    return result


async def export_to_parquet():
    """Main export function."""
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Clean up any existing parquet files
    for f in OUTPUT_DIR.glob("*.parquet"):
        f.unlink()
        logger.info(f"Removed existing file: {f}")
    
    # Initialize the database pool with longer timeout
    await db.initialize_pool(min_size=4, max_size=16, command_timeout=600)
    
    try:
        async with db.pool.acquire() as conn:
            # Get total row count
            total_rows = await get_total_row_count(conn)
            n_chunks = math.ceil(total_rows / ROWS_PER_CHUNK)
            logger.info(f"Exporting {total_rows:,} rows in {n_chunks} chunk(s) of {ROWS_PER_CHUNK:,} rows each")
            
            rows_processed = 0
            for chunk_idx in range(n_chunks):
                offset = chunk_idx * ROWS_PER_CHUNK
                query = BASE_QUERY.format(limit=ROWS_PER_CHUNK, offset=offset)
                logger.info(f"Fetching chunk {chunk_idx+1}/{n_chunks} (OFFSET {offset})")
                rows = await conn.fetch(query)
                if not rows:
                    logger.info(f"No more rows at chunk {chunk_idx+1}, stopping early.")
                    break
                columns = list(rows[0].keys()) if rows else []
                if chunk_idx == 0:
                    logger.info(f"Columns: {columns}")
                data = [dict(row) for row in rows]
                df = pd.DataFrame(data)
                import pyarrow.dataset as ds
                table = pa.Table.from_pandas(df, preserve_index=False)
                file_path = OUTPUT_DIR / f"threads-part-{chunk_idx:05d}.parquet"
                ds.write_dataset(
                    table,
                    base_dir=OUTPUT_DIR,
                    basename_template=f"threads-part-{chunk_idx:05d}.parquet",
                    format="parquet",
                    compression="snappy",
                    existing_data_behavior="overwrite_or_ignore"
                )
                file_size_mb = file_path.stat().st_size / (1024 * 1024)
                logger.info(f"Wrote {file_path.name} ({len(df):,} rows, {file_size_mb:.1f} MB)")
                rows_processed += len(df)
                logger.info(f"Processed {rows_processed:,}/{total_rows:,} rows ({rows_processed/total_rows*100:.1f}%)")
            
            logger.info(f"Export complete! {rows_processed:,} total rows exported.")
            
            # List all output files
            logger.info("\nOutput files:")
            total_size = 0
            for f in sorted(OUTPUT_DIR.glob("threads-part-*.parquet")):
                size_mb = f.stat().st_size / (1024 * 1024)
                total_size += f.stat().st_size
                logger.info(f"  {f.name}: {size_mb:.1f} MB")
            
            logger.info(f"\nTotal size: {total_size / (1024 * 1024 * 1024):.2f} GB")
    except Exception as e:
        logger.error(f"Error during export: {e}")
        raise
    finally:
        await db.close_pool()


async def main():
    """Entry point."""
    start_time = datetime.now()
    logger.info(f"Starting Parquet export at {start_time}")
    
    try:
        await export_to_parquet()
    except Exception as e:
        logger.error(f"Export failed: {e}")
        raise
    
    end_time = datetime.now()
    duration = end_time - start_time
    logger.info(f"Export completed in {duration}")


if __name__ == "__main__":
    asyncio.run(main())
