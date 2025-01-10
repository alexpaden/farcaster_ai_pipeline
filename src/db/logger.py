import os
import logging
from pathlib import Path

def setup_logging():
    """Set up logging configuration"""
    # Ensure logs directory exists
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=os.getenv('LOG_LEVEL', 'INFO'),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.getenv('LOG_FILE', 'logs/pipeline.log')),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__) 