import os
import logging

def setup_logging():
    """Set up logging configuration"""
    # Configure logging to use stderr only
    logging.basicConfig(
        level=os.getenv('LOG_LEVEL', 'INFO'),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__)