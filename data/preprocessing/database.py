"""
Database Connection and Query Module

This module handles all interactions with the PostgreSQL database containing
AIS data. It provides clean interfaces for querying, loading, and managing
ship trajectory data.

Key Features:
- Connection pooling for efficient database access
- Parameterized queries to prevent SQL injection
- Batch processing for large datasets
- Error handling and connection recovery

Author: Ship Trajectory Prediction Team
Date: 2025-11-04
"""

import logging
import psycopg2
from psycopg2 import pool, sql
from psycopg2.extras import RealDictCursor
import pandas as pd
from typing import Optional, List, Dict, Tuple
from datetime import datetime
from contextlib import contextmanager

from config.config import config

logger = logging.getLogger(__name__)


class DatabaseConnection:
    """
    Manages PostgreSQL database connections with connection pooling
    for efficient resource management.
    """
    
    def __init__(self, min_conn: int = 1, max_conn: int = 10):
        """
        Initialize database connection pool
        
        Args:
            min_conn: Minimum number of connections in pool
            max_conn: Maximum number of connections in pool
        """
        self.db_config = config.database
        self.connection_pool = None
        self.min_conn = min_conn
        self.max_conn = max_conn
        
        try:
            self.connection_pool = psycopg2.pool.SimpleConnectionPool(
                min_conn,
                max_conn,
                host=self.db_config.host,
                port=self.db_config.port,
                database=self.db_config.database,
                user=self.db_config.user,
                password=self.db_config.password
            )
            logger.info(f"Database connection pool created: {min_conn}-{max_conn} connections")
        except psycopg2.Error as e:
            logger.error(f"Failed to create connection pool: {e}")
            raise
    
    @contextmanager
    def get_connection(self):
        """
        Context manager for database connections
        Ensures connections are properly returned to pool
        
        Yields:
            psycopg2.connection: Database connection
        """
        connection = None
        try:
            connection = self.connection_pool.getconn()
            yield connection
        except psycopg2.Error as e:
            logger.error(f"Database connection error: {e}")
            if connection:
                connection.rollback()
            raise
        finally:
            if connection:
                self.connection_pool.putconn(connection)
    
    def close_all_connections(self):
        """Close all connections in the pool"""
        if self.connection_pool:
            self.connection_pool.closeall()
            logger.info("All database connections closed")


class AISDataLoader:
    """
    Handles loading and querying AIS data from the database.
    Provides methods for various query patterns needed for trajectory prediction.
    """
    
    def __init__(self, db_connection: DatabaseConnection):
        """
        Initialize AIS data loader
        
        Args:
            db_connection: DatabaseConnection instance
        """
        self.db = db_connection
        logger.info("AIS Data Loader initialized")
    
    def get_table_info(self, table_name: str = 'ais1_position') -> Dict:
        """
        Get information about the AIS data table
        
        Args:
            table_name: Name of the AIS table
            
        Returns:
            Dictionary with table statistics
        """
        with self.db.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Get row count
                cursor.execute(sql.SQL("SELECT COUNT(*) as total_records FROM {}").format(
                    sql.Identifier(table_name)
                ))
                total_records = cursor.fetchone()['total_records']
                
                # Get unique vessels
                cursor.execute(sql.SQL("SELECT COUNT(DISTINCT mmsi) as unique_vessels FROM {}").format(
                    sql.Identifier(table_name)
                ))
                unique_vessels = cursor.fetchone()['unique_vessels']
                
                # Get time range
                cursor.execute(sql.SQL(
                    "SELECT MIN(timestamp) as start_time, MAX(timestamp) as end_time FROM {}"
                ).format(sql.Identifier(table_name)))
                time_range = cursor.fetchone()
                
                info = {
                    'table_name': table_name,
                    'total_records': total_records,
                    'unique_vessels': unique_vessels,
                    'start_time': time_range['start_time'],
                    'end_time': time_range['end_time'],
                    'duration_days': (time_range['end_time'] - time_range['start_time']).days if time_range['end_time'] else 0
                }
                
                logger.info(f"Table info: {info}")
                return info
    
    def load_raw_data(self,
                      limit: Optional[int] = None,
                      mmsi_filter: Optional[List[int]] = None,
                      start_time: Optional[datetime] = None,
                      end_time: Optional[datetime] = None,
                      table_name: str = 'ais1_position') -> pd.DataFrame:
        """
        Load raw AIS data from database with optional filters
        
        Args:
            limit: Maximum number of records to load
            mmsi_filter: List of MMSI numbers to filter
            start_time: Start datetime for filtering
            end_time: End datetime for filtering
            table_name: Name of the AIS table
            
        Returns:
            DataFrame with AIS data
        """
        # Build query dynamically
        query_parts = [f"SELECT mmsi, latitude, longitude, sog, cog, timestamp FROM {table_name}"]

        conditions = []
        params = []

        if mmsi_filter:
            conditions.append("mmsi = ANY(%s)")
            params.append(mmsi_filter)

        if start_time:
            conditions.append("timestamp >= %s")
            params.append(start_time)

        if end_time:
            conditions.append("timestamp <= %s")
            params.append(end_time)

        if conditions:
            query_parts.append("WHERE " + " AND ".join(conditions))

        query_parts.append("ORDER BY mmsi, timestamp")

        if limit:
            query_parts.append(f"LIMIT {limit}")

        query = " ".join(query_parts)
        
        logger.info(f"Loading AIS data with query: {query[:200]}...")
        
        with self.db.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)
            logger.info(f"Loaded {len(df)} records from database")
            return df
    
    def load_vessel_trajectory(self,
                               mmsi: int,
                               start_time: Optional[datetime] = None,
                               end_time: Optional[datetime] = None,
                               table_name: str = 'ais1_position') -> pd.DataFrame:
        """
        Load complete trajectory for a specific vessel
        
        Args:
            mmsi: Maritime Mobile Service Identity number
            start_time: Optional start time filter
            end_time: Optional end time filter
            table_name: Name of the AIS table
            
        Returns:
            DataFrame with vessel trajectory
        """
        query = """
            SELECT mmsi, latitude, longitude, sog, cog, timestamp, 
                   vessel_type, length, width, draught, destination
            FROM {}
            WHERE mmsi = %s
        """.format(table_name)
        
        params = [mmsi]
        
        if start_time:
            query += " AND timestamp >= %s"
            params.append(start_time)
        
        if end_time:
            query += " AND timestamp <= %s"
            params.append(end_time)
        
        query += " ORDER BY timestamp"
        
        with self.db.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)
            logger.info(f"Loaded trajectory for MMSI {mmsi}: {len(df)} points")
            return df
    
    def get_vessels_in_region(self,
                              min_lat: float,
                              max_lat: float,
                              min_lon: float,
                              max_lon: float,
                              timestamp: datetime,
                              time_window_seconds: int = 300,
                              table_name: str = 'ais1_position') -> pd.DataFrame:
        """
        Get all vessels in a geographic region at a specific time
        
        Args:
            min_lat: Minimum latitude
            max_lat: Maximum latitude
            min_lon: Minimum longitude
            max_lon: Maximum longitude
            timestamp: Reference timestamp
            time_window_seconds: Time window around timestamp (±seconds)
            table_name: Name of the AIS table
            
        Returns:
            DataFrame with vessel positions in region
        """
        query = f"""
            SELECT DISTINCT ON (mmsi) 
                   mmsi, latitude, longitude, sog, cog, timestamp
            FROM {table_name}
            WHERE latitude BETWEEN %s AND %s
              AND longitude BETWEEN %s AND %s
              AND timestamp BETWEEN %s AND %s
            ORDER BY mmsi, ABS(EXTRACT(EPOCH FROM (timestamp - %s)))
        """
        
        from datetime import timedelta
        start_time = timestamp - timedelta(seconds=time_window_seconds)
        end_time = timestamp + timedelta(seconds=time_window_seconds)
        
        params = [min_lat, max_lat, min_lon, max_lon, start_time, end_time, timestamp]
        
        with self.db.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)
            logger.info(f"Found {len(df)} vessels in region at {timestamp}")
            return df
    
    def load_data_in_batches(self,
                             batch_size: int = 100000,
                             table_name: str = 'ais1_position',
                             order_by: str = 'timestamp') -> pd.DataFrame:
        """
        Load large datasets in batches to avoid memory issues
        
        Args:
            batch_size: Number of records per batch
            table_name: Name of the AIS table
            order_by: Column to order by
            
        Yields:
            DataFrame batches
        """
        offset = 0
        
        while True:
            query = f"""
                SELECT mmsi, latitude, longitude, sog, cog, timestamp
                FROM {table_name}
                ORDER BY {order_by}
                LIMIT %s OFFSET %s
            """
            
            with self.db.get_connection() as conn:
                df = pd.read_sql_query(query, conn, params=[batch_size, offset])
            
            if df.empty:
                logger.info("All data loaded")
                break
            
            logger.info(f"Loaded batch: offset={offset}, size={len(df)}")
            yield df
            
            offset += batch_size
    
    def get_data_statistics(self, table_name: str = 'ais1_position') -> Dict:
        """
        Calculate comprehensive statistics about the AIS dataset
        
        Args:
            table_name: Name of the AIS table
            
        Returns:
            Dictionary with dataset statistics
        """
        with self.db.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Basic statistics
                cursor.execute(f"""
                    SELECT 
                        COUNT(*) as total_records,
                        COUNT(DISTINCT mmsi) as unique_vessels,
                        MIN(timestamp) as earliest_record,
                        MAX(timestamp) as latest_record,
                        AVG(sog) as avg_speed,
                        MIN(sog) as min_speed,
                        MAX(sog) as max_speed,
                        COUNT(CASE WHEN sog IS NULL THEN 1 END) as null_sog,
                        COUNT(CASE WHEN cog IS NULL THEN 1 END) as null_cog,
                        COUNT(CASE WHEN latitude IS NULL THEN 1 END) as null_lat,
                        COUNT(CASE WHEN longitude IS NULL THEN 1 END) as null_lon
                    FROM {table_name}
                """)
                
                stats = dict(cursor.fetchone())
                logger.info("Dataset statistics calculated")
                return stats


def test_database_connection():
    """Test database connection and basic queries"""
    try:
        logger.info("Testing database connection...")
        
        # Initialize connection
        db = DatabaseConnection(min_conn=1, max_conn=5)
        loader = AISDataLoader(db)
        
        # Get table information
        info = loader.get_table_info()
        print("\n=== Database Table Information ===")
        for key, value in info.items():
            print(f"{key}: {value}")
        
        # Get statistics
        stats = loader.get_data_statistics()
        print("\n=== Dataset Statistics ===")
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        # Load sample data
        print("\n=== Loading Sample Data ===")
        sample_df = loader.load_raw_data(limit=10)
        print(sample_df.head())
        print(f"\nColumns: {sample_df.columns.tolist()}")
        print(f"Shape: {sample_df.shape}")
        
        # Close connections
        db.close_all_connections()
        
        logger.info("Database connection test successful!")
        return True
        
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        return False


if __name__ == "__main__":
    # Run connection test
    test_database_connection()