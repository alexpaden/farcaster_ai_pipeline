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
import time
import glob
import gc

# Add parent directory to path to import from src
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.db.connect import db
from src.db.logger import setup_logging

# Set up logging
logger = setup_logging()

# Configuration
ROWS_PER_CHUNK = 250_000      # Process this many rows at a time in memory
ROWS_PER_FILE = 2_000_000       # 100k rows per file for testing
OUTPUT_DIR = Path("data/threads")   # Output directory

# The query to execute
QUERY = """
SELECT 
    hash, 
    fids, 
    reactions, 
    author_fid, 
    "timestamp", 
    tokens,
    blob_timestamp, 
    "blob",
    blob_embedding,
    blob_embedding_binary
FROM unbias.threads
WHERE thread_status = 5 AND spam = 2
ORDER BY timestamp
"""


async def get_total_row_count(conn):
    """Get the total number of rows that will be exported."""
    count_query = """
    SELECT COUNT(*)
    FROM unbias.threads
    WHERE thread_status = 5 AND spam = 2
    """
    result = await conn.fetchval(count_query)
    return result


async def export_to_parquet():
    """Main export function with resume capability."""
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Clean up any existing parquet files that are incomplete (optional, not implemented here)
    # Scan for existing files to resume
    existing_files = sorted(OUTPUT_DIR.glob("threads-part-*.parquet"))
    if existing_files:
        last_file = existing_files[-1]
        last_file_idx = int(str(last_file.name).split('-part-')[1].split('.')[0])
        file_idx = last_file_idx + 1  # Start with the NEXT file
        rows_processed = file_idx * ROWS_PER_FILE  # Already processed up to this point
        logger.info(f"Resuming from file {last_file.name}, skipping {rows_processed:,} rows.")
        logger.info(f"Next file to write: threads-part-{file_idx:05d}.parquet")
    else:
        file_idx = 0
        rows_processed = 0
        logger.info("No existing files found, starting from the beginning.")
    rows_in_batch = 0
    batch_dfs = []
    
    # Initialize the database pool with longer timeout
    await db.initialize_pool(min_size=4, max_size=16, command_timeout=600)
    
    try:
        async with db.pool.acquire() as conn:
            # Get total row count
            total_rows = await get_total_row_count(conn)
            n_files = math.ceil(total_rows / ROWS_PER_FILE)
            logger.info(f"Exporting {total_rows:,} rows -> {n_files} parquet file(s)")
            
            # Set up cursor for streaming results
            async with conn.transaction():
                # Create a cursor that will stream results, with OFFSET for resume
                resume_query = QUERY + f" OFFSET {rows_processed}"
                cursor = await conn.cursor(resume_query)
                
                # Process rows in chunks
                while True:
                    # Fetch a chunk of rows
                    t0 = time.time()
                    rows = await cursor.fetch(ROWS_PER_CHUNK)
                    t1 = time.time()
                    logger.info(f"DB fetch took {t1-t0:.2f} seconds for {len(rows):,} rows")
                    if not rows:
                        break
                    
                    # Convert to DataFrame
                    # Extract column names from the first row if we haven't already
                    if rows_processed == 0 and file_idx == 0:
                        columns = list(rows[0].keys())
                        logger.info(f"Columns: {columns}")
                    
                    # Convert rows to list of dicts for pandas
                    t2 = time.time()
                    data = []
                    for row in rows:
                        row_dict = dict(row)
                        # Convert BitString to bytes for blob_embedding_binary
                        if 'blob_embedding_binary' in row_dict and row_dict['blob_embedding_binary'] is not None:
                            row_dict['blob_embedding_binary'] = bytes(row_dict['blob_embedding_binary'])
                        data.append(row_dict)
                    df = pd.DataFrame(data)
                    t3 = time.time()
                    logger.info(f"DataFrame construction took {t3-t2:.2f} seconds")
                    
                    batch_dfs.append(df)
                    rows_in_batch += len(df)
                    rows_processed += len(df)
                    
                    logger.info(f"Processed {rows_processed:,}/{total_rows:,} rows ({rows_processed/total_rows*100:.1f}%)")
                    
                    # Write to file when we have enough rows
                    if rows_in_batch >= ROWS_PER_FILE:
                        # Concatenate all DataFrames in the batch
                        t4 = time.time()
                        combined_df = pd.concat(batch_dfs, ignore_index=True)
                        t5 = time.time()
                        logger.info(f"DataFrame concat took {t5-t4:.2f} seconds")
                        
                        # Write to parquet
                        t6 = time.time()
                        file_path = OUTPUT_DIR / f"threads-part-{file_idx:05d}.parquet"
                        table = pa.Table.from_pandas(combined_df, preserve_index=False)
                        t7 = time.time()
                        logger.info(f"Arrow Table conversion took {t7-t6:.2f} seconds")
                        pq.write_table(
                            table,
                            file_path,
                            compression='zstd',
                            compression_level=5,
                            use_dictionary=False,
                            data_page_size=1 << 20
                        )
                        t8 = time.time()
                        logger.info(f"Parquet writing took {t8-t7:.2f} seconds")
                        
                        file_size_mb = file_path.stat().st_size / (1024 * 1024)
                        logger.info(f"Wrote {file_path.name} ({len(combined_df):,} rows, {file_size_mb:.1f} MB)")
                        
                        # Reset for next batch
                        # Explicit memory cleanup
                        del combined_df
                        del table
                        batch_dfs.clear()  # More explicit than batch_dfs = []
                        batch_dfs = []
                        rows_in_batch = 0
                        file_idx += 1
                        gc.collect()  # Force garbage collection
                        logger.info("Memory cleanup completed")
                
                # Write final partial batch if any remains
                if batch_dfs:
                    t4 = time.time()
                    combined_df = pd.concat(batch_dfs, ignore_index=True)
                    t5 = time.time()
                    logger.info(f"DataFrame concat took {t5-t4:.2f} seconds")
                    file_path = OUTPUT_DIR / f"threads-part-{file_idx:05d}.parquet"
                    t6 = time.time()
                    table = pa.Table.from_pandas(combined_df, preserve_index=False)
                    t7 = time.time()
                    logger.info(f"Arrow Table conversion took {t7-t6:.2f} seconds")
                    pq.write_table(
                        table,
                        file_path,
                        compression='zstd',
                        compression_level=5,
                        use_dictionary=False,
                        data_page_size=1 << 20
                    )
                    t8 = time.time()
                    logger.info(f"Parquet writing took {t8-t7:.2f} seconds")
                    
                    file_size_mb = file_path.stat().st_size / (1024 * 1024)
                    logger.info(f"Wrote {file_path.name} ({len(combined_df):,} rows, {file_size_mb:.1f} MB)")
                    # Explicit memory cleanup
                    del combined_df
                    del table
                    batch_dfs.clear()
                    gc.collect()
            
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
