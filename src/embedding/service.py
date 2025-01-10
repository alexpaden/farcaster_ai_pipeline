"""
FastAPI service for text embeddings with health monitoring and metrics.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import numpy as np
import time
import psutil
from .model import MPSEmbeddingModel

app = FastAPI()
model = MPSEmbeddingModel()

class EmbeddingRequest(BaseModel):
    texts: List[str]
    batch_size: int = 128

class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    metrics: Dict[str, Any]

@app.post("/embed384")
async def embed_texts(request: EmbeddingRequest) -> EmbeddingResponse:
    """Generate embeddings for a batch of texts."""
    try:
        start_time = time.time()
        
        # Generate embeddings
        embeddings = model.encode(request.texts)
        
        # Get metrics
        duration = time.time() - start_time
        texts_per_second = len(request.texts) / duration
        memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        metrics = {
            **model.get_metrics(),
            "texts_per_second": texts_per_second,
            "total_texts": len(request.texts),
            "duration_seconds": duration,
            "memory_usage_mb": memory
        }
        
        return EmbeddingResponse(
            embeddings=embeddings.tolist(),
            metrics=metrics
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    return {
        "status": "healthy",
        "memory_usage_mb": memory,
        **model.get_metrics()
    } 