import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import json
from datetime import datetime
from typing import List, Dict, Tuple
import os

def load_station_coordinates() -> Dict:
    """Load station coordinates from cache"""
    # Get the directory where Clusters.py is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Navigate to the json file
    json_path = os.path.join(current_dir, "..", "bikeshare-api", "get_clusters", "station_coordinates_cache.json")
    
    with open(json_path, "r") as f:
        return json.load(f)

def prepare_flow_data(trip_data_path: str) -> pd.DataFrame:
    """
    Prepare and aggregate flow data into 6-hour periods from raw trip data
    """
    # Load trip data
    year_data_2023 = pd.read_csv(trip_data_path)
    
    # Convert timestamps and extract components
    year_data_2023['start_time'] = pd.to_datetime(year_data_2023['Start Time'], errors='coerce')
    year_data_2023['end_time'] = pd.to_datetime(year_data_2023['End Time'], errors='coerce')
    year_data_2023['start_hour'] = year_data_2023['start_time'].dt.hour
    year_data_2023['end_hour'] = year_data_2023['end_time'].dt.hour
    year_data_2023['date'] = year_data_2023['start_time'].dt.date

    # Calculate outflow (bikes leaving)
    outflow = year_data_2023.groupby([
        'Start Station Id', 
        'Start Station Name', 
        'date', 
        'start_hour'
    ]).agg(bikes_leaving=('Trip Id', 'count')).reset_index()

    # Calculate inflow (bikes arriving)
    inflow = year_data_2023.groupby([
        'End Station Id', 
        'End Station Name', 
        'date', 
        'end_hour'
    ]).agg(bikes_arriving=('Trip Id', 'count')).reset_index()

    # Rename columns for consistency
    inflow.rename(columns={
        'End Station Id': 'station_id',
        'End Station Name': 'station_name',
        'end_hour': 'hour'
    }, inplace=True)

    outflow.rename(columns={
        'Start Station Id': 'station_id',
        'Start Station Name': 'station_name',
        'start_hour': 'hour'
    }, inplace=True)

    # Create a DataFrame for all hours, dates, and stations
    all_hours = pd.DataFrame({
        'hour': np.tile(np.arange(24), len(year_data_2023['date'].unique())),
        'date': np.repeat(year_data_2023['date'].unique(), 24)
    })

    # Generate all combinations of stations, dates, and hours
    all_combinations = year_data_2023[['Start Station Id', 'Start Station Name']].drop_duplicates()
    all_combinations.rename(columns={
        'Start Station Id': 'station_id',
        'Start Station Name': 'station_name'
    }, inplace=True)

    all_combinations = all_combinations.merge(all_hours, how='cross')

    # Merge with outflow and inflow data
    flow_data = all_combinations.merge(
        outflow, 
        on=['station_id', 'station_name', 'date', 'hour'], 
        how='left'
    ).fillna(0)
    
    flow_data = flow_data.merge(
        inflow, 
        on=['station_id', 'station_name', 'date', 'hour'], 
        how='left'
    ).fillna(0)

    # Create 6-hour periods and calculate net flow
    flow_data['period'] = flow_data['hour'] // 6
    flow_data['net_flow'] = flow_data['bikes_leaving'] - flow_data['bikes_arriving']
    
    # Aggregate by station and period
    period_flows = flow_data.groupby(
        ['station_id', 'station_name', 'period']
    )['net_flow'].mean().reset_index()
    
    return period_flows

def create_station_features(coords_data: Dict, flow_data: pd.DataFrame) -> pd.DataFrame:
    """
    Create feature matrix for clustering, combining coordinates and flow data
    """
    # Create DataFrame from coordinates
    stations_df = pd.DataFrame([
        {
            'station_name': name,
            'latitude': data['latitude'],
            'longitude': data['longitude']
        }
        for name, data in coords_data.items()
    ])
    
    # First aggregate flow data to handle duplicates
    flow_data_agg = flow_data.groupby(['station_name', 'period'])['net_flow'].mean().reset_index()
    
    # Then pivot the aggregated data
    flow_pivot = flow_data_agg.pivot(
        index='station_name',
        columns='period',
        values='net_flow'
    ).reset_index()
    
    # Merge with coordinates
    station_features = pd.merge(
        stations_df,
        flow_pivot,
        on='station_name',
        how='left'
    )
    
    # Fill NaN values with 0
    station_features = station_features.fillna(0)
    
    # Ensure we have all period columns (0,1,2,3)
    for period in range(4):
        if period not in station_features.columns:
            station_features[period] = 0
    
    return station_features

def cluster_stations(features_df: pd.DataFrame, target_size: int = 20, max_size: int = 30, min_size: int = 15) -> List[List[str]]:
    """
    Cluster stations based on location and flow patterns, ensuring all stations are assigned
    
    Args:
        features_df: DataFrame with station features
        target_size: Target number of stations per cluster
        max_size: Maximum stations per cluster
        min_size: Minimum stations per cluster
    """
    # Scale features
    scaler = StandardScaler()
    
    # Combine location and flow features with different weights
    location_features = features_df[['latitude', 'longitude']].values
    flow_features = features_df[[0, 1, 2, 3]].values  # 6-hour period columns
    
    # Scale separately
    scaled_location = scaler.fit_transform(location_features)
    scaled_flow = scaler.fit_transform(flow_features)
    
    # Combine features with weights
    combined_features = np.hstack([
        scaled_location * 0.7,
        scaled_flow * 0.3
    ])
    
    # Calculate number of clusters needed
    n_stations = len(features_df)
    n_clusters = max(2, n_stations // target_size)  # At least 2 clusters
    
    # Use KMeans for initial clustering
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    initial_clusters = kmeans.fit_predict(combined_features)
    
    # Organize stations into clusters
    clustered_stations = []
    for cluster_id in range(n_clusters):
        cluster_mask = initial_clusters == cluster_id
        cluster_stations = features_df.loc[cluster_mask, 'station_name'].tolist()
        
        # If cluster is too large, split it
        while len(cluster_stations) > max_size:
            # Create a new cluster with the furthest stations
            new_cluster = cluster_stations[max_size:]
            cluster_stations = cluster_stations[:max_size]
            if len(new_cluster) >= min_size:
                clustered_stations.append(new_cluster)
        
        # Only add clusters that meet minimum size
        if len(cluster_stations) >= min_size:
            clustered_stations.append(cluster_stations)
    
    # Handle remaining stations
    remaining_stations = []
    for cluster_id in range(n_clusters):
        cluster_mask = initial_clusters == cluster_id
        cluster_stations = features_df.loc[cluster_mask, 'station_name'].tolist()
        if len(cluster_stations) < min_size:
            remaining_stations.extend(cluster_stations)
    
    # Distribute remaining stations to nearest clusters
    if remaining_stations:
        # Sort clusters by size
        clustered_stations.sort(key=len)
        
        # Add remaining stations to smallest clusters that aren't full
        for station in remaining_stations:
            # Find station coordinates
            station_coords = features_df[features_df['station_name'] == station][['latitude', 'longitude']].values[0]
            
            # Find best cluster
            best_cluster_idx = 0
            min_avg_distance = float('inf')
            
            for i, cluster in enumerate(clustered_stations):
                if len(cluster) >= max_size:
                    continue
                    
                # Calculate average distance to cluster stations
                cluster_coords = features_df[features_df['station_name'].isin(cluster)][['latitude', 'longitude']].values
                distances = np.mean([haversine(station_coords[0], station_coords[1], 
                                            c[0], c[1]) for c in cluster_coords])
                
                if distances < min_avg_distance:
                    min_avg_distance = distances
                    best_cluster_idx = i
            
            # Add to best cluster
            clustered_stations[best_cluster_idx].append(station)
    
    return clustered_stations

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points on the Earth
    """
    R = 6371  # Earth radius in kilometers
    
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c

def validate_clusters(clusters: List[List[str]], flow_data: pd.DataFrame) -> List[List[str]]:
    """
    Validate clusters based on net flow constraints
    Returns only clusters that meet the flow balance criteria
    """
    valid_clusters = []
    
    for cluster in clusters:
        # Get flow data for stations in cluster
        cluster_flows = flow_data[flow_data['station_name'].isin(cluster)]
        
        # Calculate total net flow for each period
        period_totals = cluster_flows.groupby('period')['net_flow'].sum()
        
        # Check if all periods meet the -50 to +50 criterion
        if all((-50 <= flow <= 50) for flow in period_totals):
            valid_clusters.append(cluster)
    
    return valid_clusters

def main():
    # Load data
    coords_data = load_station_coordinates()
    
    # Process all monthly data files
    monthly_data = []
    try: 
        # Update the file path to match your actual data location
        file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            "BikeDataJanuary.csv"
        )
        monthly_flow = prepare_flow_data(file_path)
        if monthly_flow is not None:
            monthly_data.append(monthly_flow)
            print(f"Successfully loaded bike data")
    except FileNotFoundError:
        print(f"Warning: Data file not found: {file_path}")
    
    if not monthly_data:
        raise ValueError("No data files were successfully loaded. Check file paths and naming convention.")
        
    # Combine all monthly data
    flow_data = pd.concat(monthly_data).groupby(
        ['station_id', 'station_name', 'period']
    )['net_flow'].mean().reset_index()
    
    # Continue with clustering
    station_features = create_station_features(coords_data, flow_data)
    clusters = cluster_stations(station_features)
    valid_clusters = validate_clusters(clusters, flow_data)
    
    # Save results to current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, "station_clusters.json")
    
    with open(output_path, "w") as f:
        json.dump({
            "clusters": valid_clusters,
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)
    
    # Print summary
    print(f"Created {len(valid_clusters)} valid clusters")
    print(f"Total stations clustered: {sum(len(c) for c in valid_clusters)}")
    print(f"Clusters saved to: {output_path}")
    
    # Print detailed cluster information
    for i, cluster in enumerate(valid_clusters):
        print(f"\nCluster {i+1}:")
        print(f"Number of stations: {len(cluster)}")
        print("Sample stations:", cluster[:3], "...")

if __name__ == "__main__":
    main()
