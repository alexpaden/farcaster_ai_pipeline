# Threads Embedding Workflow

This workflow processes threads from the `unbias.threads` table to generate embeddings using Voyage AI's `voyage-3.5-lite` model.

## Overview

The workflow:
1. Claims batches of threads where `threads_status = 1` and `spam = 2`
2. Generates 512-dimensional embeddings using Voyage AI
3. Stores embeddings in both `blob_embedding` (vector) and `blob_embedding_fp16` (halfvec) columns
4. Updates thread status to track progress

## Prerequisites

1. Install dependencies:
```bash
pip install voyageai pgvector asyncpg numpy
```

2. Set your Voyage AI API key:
```bash
export VOYAGE_API_KEY="your-api-key-here"
```

## Status Codes

- `0`: Unclaimed (older, unused)
- `1`: Pending embed
- `2`: Claimed (locked by a worker)
- `3`: Embedded (complete)
- `4`: Failed (API error, timeout, etc.)

## Usage

### Test Mode (Process 10 rows max)
```bash
python -m src.threads.workflow --test
```

### Basic Usage
```bash
# Process one batch of 128 texts
python -m src.threads.workflow

# Process multiple batches
python -m src.threads.workflow --max-batches 10

# Use multiple workers
python -m src.threads.workflow --workers 4 --max-batches 100
```

### Command Line Options

- `--batch-size`: Number of texts per batch (default: 128, max: 1000)
- `--max-batches`: Maximum batches per worker (default: 1)
- `--workers`: Number of parallel workers (default: 1)
- `--test`: Test mode - process max 10 rows

### Examples

```bash
# Production run with 8 workers, 100 batches each
python -m src.threads.workflow --workers 8 --max-batches 100

# Custom batch size (conservative)
python -m src.threads.workflow --batch-size 64 --workers 4

# Larger batch size for higher throughput (monitor for token limits)
python -m src.threads.workflow --batch-size 1000 --workers 3 --max-batches 0

# Test run
python -m src.threads.workflow --test --batch-size 5
```

## Technical Details

### Text Processing
- Replaces "↳" with "-" in memory before embedding (reduces token usage)
- Handles empty/null texts gracefully
- Uses `input_type="document"` for optimal retrieval performance

### Embedding Configuration
- Uses `voyage-3.5-lite` model with 512-dimensional output
- Specifies `output_dimension=512` to match database schema
- Database columns: `VECTOR(512)` and `HALFVEC(512)`
- Note: voyage-3.5-lite defaults to 1024 dimensions if not specified

### Database Operations
- Uses `FOR UPDATE SKIP LOCKED` for parallel processing
- Claim is recognized by status change from 1 → 2
- Bulk updates using pgvector types for efficiency

### Error Handling
- Retries API calls with exponential backoff for rate limits
- Marks failed items with status 4
- Comprehensive logging for debugging

### Performance
- Batch size of 128 is a conservative choice (Voyage allows up to 1,000 texts/request)
- Chosen for safety: 128 texts × ~8k tokens ≈ well under 1M token limit
- Fits within 5-minute DB command timeout
- Leaves room for retry/backoff on HTTP 429 rate limits
- You can increase batch size up to 1,000 if monitoring token usage
- Parallel workers can process ~25M rows efficiently
- Automatic float32 to float16 conversion for storage efficiency

## Monitoring

Watch the logs for progress:
```
2024-01-01 12:00:00 - Worker-0 - INFO - Starting worker 0
2024-01-01 12:00:01 - Worker-0 - INFO - Claimed batch of 128 items
2024-01-01 12:00:02 - Worker-0 - INFO - Generating embeddings for 128 texts (attempt 1/3)
2024-01-01 12:00:05 - Worker-0 - INFO - Successfully generated 128 embeddings
2024-01-01 12:00:06 - Worker-0 - INFO - Updated 128 embeddings in database
```

## SQL Queries for Monitoring

Check progress:
```sql
SELECT threads_status, COUNT(*) 
FROM unbias.threads 
WHERE spam = 2 
GROUP BY threads_status;
```

Find stuck items (workers that may have crashed):
```sql
-- Items claimed for over an hour might indicate crashed workers
SELECT COUNT(*) as stuck_count
FROM unbias.threads 
WHERE threads_status = 2;
```

Reset stuck items if needed:
```sql
-- Manually reset items back to pending if workers crashed
UPDATE unbias.threads 
SET threads_status = 1
WHERE threads_status = 2;
``` 