"""
Utility functions for the embedding pipeline.
"""

import os
import psutil
import subprocess
import traceback
from typing import Tuple

def get_memory_usage() -> Tuple[float, float]:
    """Get current process and GPU memory usage in MB."""
    try:
        # Get process memory
        process = psutil.Process(os.getpid())
        process_mem = process.memory_info().rss / (1024 * 1024)
        
        # Get GPU memory (MPS)
        try:
            result = subprocess.run(['ps', '-o', 'rss=', '-p', str(os.getpid())], 
                                  capture_output=True, text=True)
            gpu_mem = int(result.stdout.strip()) / 1024
        except:
            gpu_mem = 0
            
        return process_mem, gpu_mem
    except:
        return 0.0, 0.0

def log_and_reraise(e: Exception, context: str):
    """Log an exception with context and re-raise it."""
    print(f"\nError in {context}:")
    print(f"Error type: {type(e).__name__}")
    print(f"Error details: {str(e)}")
    print("Traceback:")
    print(traceback.format_exc())
    raise

def format_number(n: float, decimals: int = 1) -> str:
    """Format a number with thousands separator and fixed decimals."""
    return f"{n:,.{decimals}f}" 