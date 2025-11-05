"""
Generate Vessel Registry

Creates a CSV file with all vessel IDs (MMSI) from the database along with
basic statistics about each vessel's trajectory data.

Usage:
    python3 utils/vessel_registry.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
from data.preprocessing.database import DatabaseConnection, AISDataLoader
from datetime import datetime

def generate_vessel_registry():
    """Generate a comprehensive vessel registry from the database"""
    
    print("Connecting to database...")
    db = DatabaseConnection(min_conn=1, max_conn=2)
    loader = AISDataLoader(db)
    
    print("Querying vessel statistics...")
    
    # Get comprehensive vessel statistics
    query = """
    SELECT 
        mmsi,
        COUNT(*) as total_records,
        MIN(timestamp) as first_seen,
        MAX(timestamp) as last_seen,
        EXTRACT(EPOCH FROM (MAX(timestamp) - MIN(timestamp)))/3600 as duration_hours,
        AVG(latitude) as avg_latitude,
        AVG(longitude) as avg_longitude,
        AVG(sog) as avg_speed_knots,
        MIN(sog) as min_speed_knots,
        MAX(sog) as max_speed_knots,
        AVG(cog) as avg_course
    FROM ais1_position
    GROUP BY mmsi
    ORDER BY total_records DESC;
    """
    
    with db.get_connection() as conn:
        df = pd.read_sql_query(query, conn)

    # Format the data
    df['first_seen'] = pd.to_datetime(df['first_seen'])
    df['last_seen'] = pd.to_datetime(df['last_seen'])
    df['duration_days'] = df['duration_hours'] / 24

    # Round numeric columns
    df['avg_latitude'] = df['avg_latitude'].round(4)
    df['avg_longitude'] = df['avg_longitude'].round(4)
    df['avg_speed_knots'] = df['avg_speed_knots'].round(2)
    df['min_speed_knots'] = df['min_speed_knots'].round(2)
    df['max_speed_knots'] = df['max_speed_knots'].round(2)
    df['avg_course'] = df['avg_course'].round(1)
    df['duration_days'] = df['duration_days'].round(2)

    # Reorder columns
    df = df[['mmsi', 'total_records', 'first_seen', 'last_seen', 'duration_days',
             'avg_latitude', 'avg_longitude', 'avg_speed_knots', 'min_speed_knots',
             'max_speed_knots', 'avg_course']]

    # Save to CSV
    output_file = 'utils/vessel_registry.csv'
    df.to_csv(output_file, index=False)

    print(f"\n✓ Vessel registry saved to: {output_file}")
    print(f"\nRegistry Statistics:")
    print(f"  Total Vessels: {len(df)}")
    print(f"  Total Records: {df['total_records'].sum():,}")
    print(f"  Records per Vessel:")
    print(f"    Mean:   {df['total_records'].mean():,.0f}")
    print(f"    Median: {df['total_records'].median():,.0f}")
    print(f"    Min:    {df['total_records'].min():,}")
    print(f"    Max:    {df['total_records'].max():,}")

    print(f"\n  Top 10 Vessels by Record Count:")
    print(df[['mmsi', 'total_records', 'duration_days']].head(10).to_string(index=False))

    # Also create a simple MMSI-only file for quick reference
    mmsi_only_file = 'utils/vessel_mmsi_list.txt'
    with open(mmsi_only_file, 'w') as f:
        f.write("# Vessel MMSI List\n")
        f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Total Vessels: {len(df)}\n\n")
        for mmsi in df['mmsi']:
            f.write(f"{mmsi}\n")

    print(f"\n✓ MMSI list saved to: {mmsi_only_file}")

    db.close_all_connections()
    
    print("\n✓ Vessel registry generation complete!")

if __name__ == '__main__':
    generate_vessel_registry()

