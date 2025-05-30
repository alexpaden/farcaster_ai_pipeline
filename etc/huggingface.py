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
ROWS_PER_CHUNK = 100_000      # Process this many rows at a time in memory
ROWS_PER_FILE = 5_000_000    # 10M rows per file as requested
OUTPUT_DIR = Path("../../data")   # Output directory

# The query to execute
QUERY = """
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
            n_files = math.ceil(total_rows / ROWS_PER_FILE)
            logger.info(f"Exporting {total_rows:,} rows -> {n_files} parquet file(s)")
            
            # Set up cursor for streaming results
            async with conn.transaction():
                # Create a cursor that will stream results
                cursor = await conn.cursor(QUERY)
                
                batch_dfs = []
                rows_in_batch = 0
                file_idx = 0
                rows_processed = 0
                
                # Process rows in chunks
                while True:
                    # Fetch a chunk of rows
                    rows = await cursor.fetch(ROWS_PER_CHUNK)
                    if not rows:
                        break
                    
                    # Convert to DataFrame
                    # Extract column names from the first row if we haven't already
                    if rows_processed == 0:
                        columns = list(rows[0].keys())
                        logger.info(f"Columns: {columns}")
                    
                    # Convert rows to list of dicts for pandas
                    data = [dict(row) for row in rows]
                    df = pd.DataFrame(data)
                    
                    batch_dfs.append(df)
                    rows_in_batch += len(df)
                    rows_processed += len(df)
                    
                    logger.info(f"Processed {rows_processed:,}/{total_rows:,} rows ({rows_processed/total_rows*100:.1f}%)")
                    
                    # Write to file when we have enough rows
                    if rows_in_batch >= ROWS_PER_FILE:
                        # Concatenate all DataFrames in the batch
                        combined_df = pd.concat(batch_dfs, ignore_index=True)
                        
                        # Write to parquet
                        file_path = OUTPUT_DIR / f"threads-part-{file_idx:05d}.parquet"
                        table = pa.Table.from_pandas(combined_df, preserve_index=False)
                        pq.write_table(
                            table,
                            file_path,
                            compression='snappy',
                            use_dictionary=True,
                            compression_level=None
                        )
                        
                        file_size_mb = file_path.stat().st_size / (1024 * 1024)
                        logger.info(f"Wrote {file_path.name} ({len(combined_df):,} rows, {file_size_mb:.1f} MB)")
                        
                        # Reset for next batch
                        batch_dfs = []
                        rows_in_batch = 0
                        file_idx += 1
                
                # Write final partial batch if any remains
                if batch_dfs:
                    combined_df = pd.concat(batch_dfs, ignore_index=True)
                    file_path = OUTPUT_DIR / f"threads-part-{file_idx:05d}.parquet"
                    table = pa.Table.from_pandas(combined_df, preserve_index=False)
                    pq.write_table(
                        table,
                        file_path,
                        compression='snappy',
                        use_dictionary=True,
                        compression_level=None
                    )
                    
                    file_size_mb = file_path.stat().st_size / (1024 * 1024)
                    logger.info(f"Wrote {file_path.name} ({len(combined_df):,} rows, {file_size_mb:.1f} MB)")
            
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
