"""
Database connection management with connection pooling and migrations.
"""

import os
import logging
import psycopg2
import asyncpg
from psycopg2.extras import DictCursor
from dotenv import load_dotenv
from contextlib import contextmanager
from typing import List, Dict, Any
from pathlib import Path
import glob
import re
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
        self._pool = None
        self._test_connection()

    async def run_migrations(self, module_path: str = None):
        """Run migrations in order, optionally for a specific module."""
        print("\nRunning migrations...")
        
        # Get all migration files
        migration_paths = []
        
        # Global migrations first
        global_migrations = sorted(glob.glob(str(Path(__file__).parent / "migrations" / "*.sql")))
        migration_paths.extend(global_migrations)
        
        # Module-specific migrations if specified
        if module_path:
            module_migrations = sorted(glob.glob(str(Path(module_path) / "migrations" / "*.sql")))
            migration_paths.extend(module_migrations)
        
        # Create migrations table if it doesn't exist
        async with self.pool.acquire() as conn:
           
            # Set search path to ensure migrations table is created in unbias schema
            await conn.execute("SET search_path TO unbias;")
            
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS migrations (
                    id SERIAL PRIMARY KEY,
                    filename TEXT NOT NULL UNIQUE,
                    module TEXT,
                    applied_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Get already applied migrations
            applied = set(await conn.fetch("SELECT filename FROM migrations"))
            applied = {row['filename'] for row in applied}
            
            # Apply new migrations in order
            for path in migration_paths:
                filename = os.path.basename(path)
                if filename not in applied:
                    print(f"Applying migration: {filename}")
                    
                    # Read and execute migration
                    with open(path, 'r') as f:
                        original_sql = f.read()
                        # Strip "CONCURRENTLY" from CREATE INDEX statements
                        modified_sql = re.sub(r"CREATE\s+(UNIQUE\s+)?INDEX\s+CONCURRENTLY", 
                                              r"CREATE \1INDEX", 
                                              original_sql, 
                                              flags=re.IGNORECASE)
                        
                        if original_sql != modified_sql:
                            print(f"  INFO: Removed CONCURRENTLY from index creation in {filename}")
                        
                        await conn.execute(modified_sql)
                    
                    # Record migration
                    module = os.path.basename(os.path.dirname(os.path.dirname(path))) if module_path else None
                    await conn.execute(
                        "INSERT INTO migrations (filename, module) VALUES ($1, $2)",
                        filename, module
                    )
                    print(f"Applied migration: {filename}")
        
        print("Migrations complete.")

    def _test_connection(self):
        """Test the database connection with the provided parameters."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute('SELECT 1')
            #logger.info("Database connection test successful")
        except Exception as e:
            logger.error(f"Database connection test failed: {str(e)}")
            raise

    @contextmanager
    def get_connection(self):
        """Get a synchronous database connection with error handling."""
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
        """Get a synchronous database cursor with error handling."""
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

    async def initialize_pool(self, command_timeout=120):
        """Initialize the asyncpg connection pool."""
        if self._pool is None:
            try:
                self._pool = await asyncpg.create_pool(
                    database=self.db_params['dbname'],
                    user=self.db_params['user'],
                    password=self.db_params['password'],
                    host=self.db_params['host'],
                    port=self.db_params['port'],
                    min_size=4,  # Minimum connections per process
                    max_size=8,  # Maximum connections per process
                    command_timeout=command_timeout,
                    server_settings={
                        'application_name': f'farcaster_ai_pipeline_{os.getpid()}',
                        'search_path': 'unbias,farcaster,nindexer'
                    }
                )
                logger.debug("Created new asyncpg connection pool")
            except Exception as e:
                logger.error(f"Error creating asyncpg pool: {str(e)}")
                raise
        return self._pool

    @property
    def pool(self):
        """Get the asyncpg connection pool."""
        if self._pool is None:
            raise RuntimeError("Pool not initialized. Call initialize_pool() first.")
        return self._pool

    async def get_pool(self, command_timeout=120):
        """Get or create an asyncpg connection pool."""
        if self._pool is None:
            await self.initialize_pool(command_timeout=command_timeout)
        return self._pool

    async def close_pool(self):
        """Close the asyncpg connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            logger.debug("Closed asyncpg connection pool")

    def close(self):
        """Close all database connections."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            logger.debug("Closed synchronous database connection")

# Create a singleton instance
db = DatabaseConnection() 