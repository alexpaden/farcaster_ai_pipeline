"""
Embedding pipeline for processing and storing text embeddings.
"""

from .model import OptimizedEmbeddingModel
from .batch import BatchProcessor, EmbeddingProcessor
from .logger import PipelineLogger, InferenceMetrics
from .db import get_unprocessed_estimate, db_update_worker
from .utils import get_memory_usage, log_and_reraise, format_number

__all__ = [
    'OptimizedEmbeddingModel',
    'BatchProcessor',
    'EmbeddingProcessor',
    'PipelineLogger',
    'InferenceMetrics',
    'get_unprocessed_estimate',
    'db_update_worker',
    'get_memory_usage',
    'log_and_reraise',
    'format_number'
] 