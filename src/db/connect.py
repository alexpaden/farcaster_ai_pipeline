import os
import logging
import psycopg2
from psycopg2.extras import DictCursor
from dotenv import load_dotenv
from contextlib import contextmanager
from .logger import setup_logging

# Load environment variables
load_dotenv()

# Set up logging
logger = setup_logging()

class DatabaseConnection:
    def __init__(self):
        self.db_params = {
            'dbname': os.getenv('DB_NAME'),
            'user': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASSWORD'),
            'host': os.getenv('DB_HOST'),
            'port': os.getenv('DB_PORT')
        }
        self._conn = None
        self._test_connection()

    def _test_connection(self):
        """Test the database connection with the provided parameters."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute('SELECT 1')
            logger.info("Database connection test successful")
        except Exception as e:
            logger.error(f"Database connection test failed: {str(e)}")
            raise

    @contextmanager
    def get_connection(self):
        """Get a database connection with error handling."""
        if self._conn is None:
            try:
                self._conn = psycopg2.connect(**self.db_params)
                logger.debug("Created new database connection")
            except Exception as e:
                logger.error(f"Error connecting to database: {str(e)}")
                raise

        try:
            yield self._conn
        except Exception as e:
            logger.error(f"Error during database operation: {str(e)}")
            if self._conn:
                self._conn.rollback()
            raise
        finally:
            if self._conn and self._conn.closed:
                self._conn = None

    @contextmanager
    def get_cursor(self, cursor_factory=DictCursor):
        """Get a database cursor with error handling."""
        with self.get_connection() as conn:
            cursor = conn.cursor(cursor_factory=cursor_factory)
            try:
                yield cursor
                conn.commit()
            except Exception as e:
                conn.rollback()
                logger.error(f"Error during cursor operation: {str(e)}")
                raise
            finally:
                cursor.close()

    def close(self):
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            logger.debug("Closed database connection")

# Create a singleton instance
db = DatabaseConnection() 