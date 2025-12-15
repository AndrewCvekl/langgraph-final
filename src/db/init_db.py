"""Database initialization for Chinook SQLite database.

Downloads the Chinook SQL schema and data from GitHub and initializes
a persistent SQLite database file.
"""

import os
import sqlite3
from pathlib import Path

import requests
from langchain_community.utilities.sql_database import SQLDatabase
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

# Database configuration
DB_DIR = Path(__file__).parent.parent.parent / "data"
DB_PATH = DB_DIR / "chinook_demo.db"
CHINOOK_SQL_URL = "https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql"

# Singleton instances
_engine = None
_db = None


def initialize_database(force: bool = False) -> Path:
    """Download and initialize the Chinook database if it doesn't exist.
    
    Args:
        force: If True, recreate the database even if it exists.
        
    Returns:
        Path to the database file.
    """
    # Create data directory if needed
    DB_DIR.mkdir(parents=True, exist_ok=True)
    
    if DB_PATH.exists() and not force:
        print(f"Database already exists at {DB_PATH}")
        return DB_PATH
    
    print(f"Downloading Chinook database from {CHINOOK_SQL_URL}...")
    response = requests.get(CHINOOK_SQL_URL, timeout=30)
    response.raise_for_status()
    sql_script = response.text
    
    # Remove existing database if forcing recreation
    if DB_PATH.exists():
        DB_PATH.unlink()
    
    print(f"Initializing database at {DB_PATH}...")
    connection = sqlite3.connect(str(DB_PATH))
    try:
        connection.executescript(sql_script)
        connection.commit()
        print("Database initialized successfully!")
    finally:
        connection.close()
    
    return DB_PATH


def get_engine():
    """Get or create the SQLAlchemy engine for the Chinook database.
    
    Returns:
        SQLAlchemy Engine instance.
    """
    global _engine
    
    if _engine is None:
        # Initialize database if it doesn't exist
        initialize_database()
        
        # Create engine with proper configuration for SQLite
        _engine = create_engine(
            f"sqlite:///{DB_PATH}",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
    
    return _engine


def get_db() -> SQLDatabase:
    """Get or create the LangChain SQLDatabase wrapper.
    
    Returns:
        SQLDatabase instance for use with LangChain tools.
    """
    global _db
    
    if _db is None:
        engine = get_engine()
        _db = SQLDatabase(engine)
    
    return _db


# Demo customer ID - in a real app this would come from authentication
DEMO_CUSTOMER_ID = int(os.environ.get("DEMO_CUSTOMER_ID", "1"))


if __name__ == "__main__":
    # Allow running this module directly to initialize the database
    initialize_database(force=True)
    db = get_db()
    print(f"Available tables: {db.get_usable_table_names()}")


