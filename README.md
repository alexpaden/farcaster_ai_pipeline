# Farcaster AI Pipeline

High-performance embedding generation pipeline for Farcaster casts using optimized MiniLM-L6-v2.

## Model Details

- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Embedding Size: 384 dimensions
- Quantization: int8 (production storage)
- Performance: ~85-90% of OpenAI ada-002 for semantic search tasks
- Batch Size: 256 (optimized)
- Instances: Dynamic scaling up to 48 (production configuration)

## Performance Metrics

- Processing Speed: ~75k texts/second with dynamic scaling
- Memory Usage: ~1.3GB per instance
- Total Memory: ~62GB at full load (48 instances)
- Estimated Processing Time for 200M casts: ~45 minutes

## Requirements

- PostgreSQL 15+ with pgvector extension
- Python 3.9+
- PyTorch 2.0+
- Apple Silicon Mac (M1/M2/M3) for MPS acceleration

## Setup

1. Create virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Set up database:
```bash
psql -f sql/schema.sql
```

3. Configure environment:
```bash
cp .env.example .env
# Edit .env with your database credentials
```

## Usage

### Run Pipeline
```bash
# Run migrations only
python src/main.py --migrate-only

# Run full pipeline
python src/main.py

# Run in test mode
python src/main.py --test

# Resume from specific cast ID
python src/main.py --start-id <cast_id>
```

## Database Schema

- Test Table: `public.test_casts`
  - For benchmarking and testing
  - 100k sample casts
  - IVF index with 100 lists
  - Includes embedding column (384d int8)

- Production Table: `public.casts`
  - Full cast history
  - 384d int8 embeddings
  - IVF index with 1000 lists
  - Optimized for high-throughput updates

## Architecture

- Modular SQL operations in `sql/` directory
- Dynamic instance scaling based on throughput
- Warm-up phase for optimal performance
- Batched database operations
- Efficient memory management

## Performance Tuning

- Dynamic Instance Scaling: 1-48 instances
- Batch Size: 256 (optimal for MPS)
- Memory per Instance: ~1.3GB
- Database Indexing: IVF for fast similarity search
- Warm-up phase for consistent performance

## Use Cases

1. In-thread Semantic Search
2. Reply Classification
3. Cast Clustering
4. Similar Cast Discovery

## Monitoring

Monitor memory usage with Activity Monitor:
- Peak Memory: ~1.3GB per instance under load
- Total Memory: ~62GB at max instances (48)
- GPU Memory: Managed by MPS
- Dynamic scaling based on throughput metrics