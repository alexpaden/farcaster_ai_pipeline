# Farcaster AI Pipeline

High-performance embedding generation pipeline for Farcaster casts using optimized MiniLM-L6-v2.

## Architecture

### Core Modules (`src/embedding384/`)

- `model.py` - Optimized MiniLM-L6-v2 with TorchScript, float16, and buffer pre-allocation
- `pipeline.py` - Dynamic scaling pipeline with workload-based instance management
- `benchmark.py` - Performance testing and optimization framework

### Database (`src/db/`)

- PostgreSQL with pgvector extension
- Optimized connection pooling and transaction management
- Efficient batch operations with SKIP LOCKED
- Error tracking and automatic retries

## Features

- Dynamic scaling from 1-48 instances based on workload (1 instance per 10k texts)
- TorchScript optimization with float16 precision on MPS
- Pre-allocated buffers for minimal memory overhead
- Fixed 256 batch size (optimized for M-series chips)
- Efficient connection pooling with asyncpg

## Performance

- Processing Speed: ~50k texts/second with 48 instances
- Memory Usage: ~2GB per instance
- Total Memory: ~96GB at full load (48 instances)
- Batch Size: 256 (optimized for MPS)
- Estimated Processing Time for 200M casts: ~67 minutes

## Requirements

- PostgreSQL 15+ with pgvector extension
- Python 3.9+
- PyTorch 2.0+ with MPS support
- Apple Silicon Mac (M1/M2/M3)
- 128GB RAM recommended for full scaling

## Setup

1. Create virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Configure environment:
```bash
cp .env.example .env
# Edit .env with your database credentials
```

3. Run migrations:
```bash
python src/main.py --migrate-only
```

## Usage

### Production Pipeline
```bash
# Run full pipeline with dynamic scaling
python src/main.py

# Run benchmark tests
python -m src.embedding384.benchmark
```

### Benchmark Results

Sample benchmark output with 256 batch size:
```
Instances: 1,  TPS: 1.2k,  Memory: 2.1GB
Instances: 6,  TPS: 7.1k,  Memory: 12.6GB
Instances: 36, TPS: 42.3k, Memory: 75.6GB
```

## Database Schema

### Production Table (`public.casts`)
```sql
CREATE TABLE casts (
    cast_id BIGINT PRIMARY KEY,
    text TEXT,
    embedding vector(384),
    embedding_updated_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_casts_unprocessed ON casts (cast_id) 
WHERE embedding IS NULL AND text IS NOT NULL;
```

### Test Table (`public.test_casts`)
- Mirror of production schema
- Used for benchmarking and testing
- Smaller dataset (100k rows)

## Monitoring

- Dynamic scaling events
- Memory usage per instance
- Processing throughput
- Database operation latencies
- Error tracking and retries

## Architecture Notes

- Optimized for Apple Silicon with MPS backend
- Efficient memory management with pre-allocated buffers
- Dynamic scaling based on unprocessed workload
- Safe database operations with SKIP LOCKED
- Clean separation of concerns between modules