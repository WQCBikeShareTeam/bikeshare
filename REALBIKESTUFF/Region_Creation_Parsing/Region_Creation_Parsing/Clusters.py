import pandas as pd
import warnings

# Suppress the specific pandas warning about setting values on a copy
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import json
from datetime import datetime
from typing import List, Dict, Tuple
import os
from sklearn.cluster import KMeans
from geopy.distance import geodesic
from concurrent.futures import ProcessPoolExecutor, as_completed  # Added for concurrency

def load_station_coordinates() -> Dict:
    """Load station coordinates from cache"""
    # Get the directory where Clusters.py is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Navigate to the json file
    json_path = os.path.join(current_dir, "..", "bikeshare-api", "get_clusters", "station_coordinates_cache.json")
    
    with open(json_path, "r") as f:
        return json.load(f)

def prepare_flow_data(trip_data_path: str, month: int = None, day_of_week: str = None, period: str = 'AM') -> pd.DataFrame:
    """
    Prepare and aggregate flow data for specific time periods
    
    Args:
        trip_data_path (str): Path to the trip data CSV
        month (int): Month number (1-12), None for all months
        day_of_week (str): Day of week ('Monday', 'Tuesday', etc.), None for all days
        period (str): 'AM' or 'PM'
    """
    # Load trip data with encoding specified
    year_data_2023 = pd.read_csv(trip_data_path, encoding='latin1')
    
    # Convert timestamps and extract components
    year_data_2023['start_time'] = pd.to_datetime(year_data_2023['Start Time'], errors='coerce')
    year_data_2023['end_time'] = pd.to_datetime(year_data_2023['End Time'], errors='coerce')
    
    # Extract all time components
    year_data_2023['month'] = year_data_2023['start_time'].dt.month
    year_data_2023['day_of_week'] = year_data_2023['start_time'].dt.day_name()
    year_data_2023['start_hour'] = year_data_2023['start_time'].dt.hour
    year_data_2023['end_hour'] = year_data_2023['end_time'].dt.hour
    year_data_2023['date'] = year_data_2023['start_time'].dt.date

    # Apply filters
    if month is not None:
        year_data_2023 = year_data_2023[year_data_2023['month'] == month]
        print(f"Filtered for month: {month}")
    
    if day_of_week is not None:
        year_data_2023 = year_data_2023[year_data_2023['day_of_week'] == day_of_week]
        print(f"Filtered for day: {day_of_week}")
    
    # Define period hours
    if period == 'AM':
        hour_range = range(0, 12)
        print("Using AM period (0-11 hours)")
    else:  # PM
        hour_range = range(12, 24)
        print("Using PM period (12-23 hours)")
    
    year_data_2023 = year_data_2023[year_data_2023['start_hour'].isin(hour_range)]

    # Calculate outflow (bikes leaving)
    outflow = year_data_2023.groupby([
        'Start Station Id', 
        'Start Station Name', 
        'date'
    ]).size().reset_index(name='bikes_leaving')

    # Calculate inflow (bikes arriving)
    inflow = year_data_2023.groupby([
        'End Station Id', 
        'End Station Name', 
        'date'
    ]).size().reset_index(name='bikes_arriving')

    # Rename columns for consistency
    inflow.rename(columns={
        'End Station Id': 'station_id',
        'End Station Name': 'station_name'
    }, inplace=True)

    outflow.rename(columns={
        'Start Station Id': 'station_id',
        'Start Station Name': 'station_name'
    }, inplace=True)

    # Merge inflow and outflow
    flow_data = pd.merge(
        outflow,
        inflow,
        on=['station_id', 'station_name', 'date'],
        how='outer'
    ).fillna(0)

    # Calculate net flow
    flow_data['net_flow'] = flow_data['bikes_arriving'] - flow_data['bikes_leaving']

    # Calculate average net flow for each station
    station_flows = flow_data.groupby(
        ['station_id', 'station_name']
    )['net_flow'].mean().reset_index()

    # Add period column for compatibility with rest of code
    station_flows['period'] = 0 if period == 'AM' else 1

    return station_flows

def calculate_net_flow(station_data):
    """Calculate net flow for a station using the available period data"""
    # Since we're only working with one period, just get the net_flow directly
    return station_data['net_flow']

def get_manual_clusters() -> Dict[str, List[str]]:
    """Define manual clusters for specific stations"""
    manual_clusters = {
        "vaughan": [
            # "Dufferin St / Finch Hydro Recreational Trail",  # Need correct name
            "Sentinel Rd / Finch Hydro Corridor",
            # "Finch West Subway Station",  # Need correct name
            "The Pond Rd / Sentinel Rd",
            "York University Station (South) - SMART",
            "York University Station (North)",
            # "G Ross Lord Park",  # Need correct name
            "Torresdale Ave / Antibes Dr",
            # "Esther Shiner Stadium",  # Need correct name
            # "Humber College"  # Need correct name
        ],
        "scarborough": [
            # "Livingston Rd / Guildwood Pkwy",  # Need correct name
            "Antler St / Campbell Ave - SMART",
            # "Guildwood GO Station (South)",  # Need correct name
            "Waterfront Trail (Rouge Hill)",
            "Rouge Hill GO Station"
        ]
    }
    return manual_clusters

def create_station_features(coords_data: Dict, flow_data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Create feature matrix for clustering, combining coordinates and flow data"""
    # Get manual clusters
    manual_clusters = get_manual_clusters()
    manual_stations = [station for cluster in manual_clusters.values() for station in cluster]
    
    # Create DataFrame from coordinates, excluding manual cluster stations
    stations_df = pd.DataFrame([
        {
            'station_name': name,
            'latitude': data['latitude'],
            'longitude': data['longitude']
        }
        for name, data in coords_data.items()
        if name not in manual_stations
    ])
    
    # Merge with flow data
    station_features = pd.merge(
        stations_df,
        flow_data[['station_name', 'net_flow']],
        on='station_name',
        how='left'
    )
    
    # Fill NaN values with 0
    station_features = station_features.fillna(0)
    
    return station_features, manual_stations

def dynamic_kmeans(station_data, manual_stations, min_flow=-100, max_flow=100, min_bikes=15, max_bikes=35, min_clusters=10, max_clusters=50, random_seed=None):
    """
    Dynamic K-Means clustering with much stricter flow controls.
    """
    MAX_ITERATIONS = 100
    
    coords = station_data[['latitude', 'longitude']].values
    station_data['net_flow'] = station_data.apply(calculate_net_flow, axis=1)

    total_stations = len(station_data)
    target_cluster_size = (min_bikes + max_bikes) // 2
    optimal_clusters = max(min_clusters, min(max_clusters, total_stations // target_cluster_size))
    
    print(f"Initializing clustering with {optimal_clusters} clusters...")
    print(f"Total stations: {total_stations}")
    print(f"Target bikes per cluster: {target_cluster_size}")

    # Initialize clustering
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=random_seed if random_seed is not None else 42, max_iter=1000, n_init=10)
    station_data['cluster'] = kmeans.fit_predict(coords)
    centroids = kmeans.cluster_centers_

    def find_best_cluster_for_station(station, current_cluster_id, cluster_flows, cluster_sizes, strict=True):
        """Find the best cluster to move a station to with much stricter flow controls"""
        best_cluster = None
        min_violation = float('inf')
        
        current_size = cluster_sizes.get(current_cluster_id, 0)
        current_flow = cluster_flows.get(current_cluster_id, 0)
        
        # Don't allow moving if it would create an undersized cluster
        if current_size <= min_bikes:
            return None
            
        # Prioritize fixing flow violations
        current_flow_violation = max(0, abs(current_flow) - max_flow)
        
        for other_cluster in range(optimal_clusters):
            if other_cluster == current_cluster_id:
                continue
                
            other_size = cluster_sizes.get(other_cluster, 0)
            other_flow = cluster_flows.get(other_cluster, 0)
            
            if other_size >= max_bikes:
                continue
                
            new_flow = other_flow + station['net_flow']
            new_remaining_flow = current_flow - station['net_flow']
            
            # Extremely strict flow controls
            if abs(new_flow) > max_flow * 0.9:  # Reduced threshold to 90% of max
                continue
                
            # Calculate if the move improves overall flow balance
            current_violation = abs(current_flow) + abs(other_flow)
            new_violation = abs(new_remaining_flow) + abs(new_flow)
            
            if new_violation >= current_violation and strict:
                continue
                
            # Calculate distance penalty
            dist = geodesic(
                (station['latitude'], station['longitude']),
                (centroids[int(other_cluster)][0], centroids[int(other_cluster)][1])
            ).meters
            
            # Modified violation score heavily weighted towards flow improvement
            violation_score = (new_violation * 4) + (dist/1000)
            
            if violation_score < min_violation:
                min_violation = violation_score
                best_cluster = other_cluster
        
        return best_cluster

    # Balance clusters with stricter controls
    MAX_ATTEMPTS_PER_CLUSTER = 6000  # Increased attempts
    
    for iteration in range(20):  # Increased iterations
        print(f"\nIteration {iteration + 1}")
        
        cluster_flows = station_data.groupby('cluster')['net_flow'].sum().to_dict()
        cluster_sizes = station_data.groupby('cluster').size().to_dict()
        
        # First, handle any undersized clusters
        undersized_clusters = [(cid, size) for cid, size in cluster_sizes.items() if size < min_bikes]
        if undersized_clusters:
            print(f"Warning: Found {len(undersized_clusters)} undersized clusters")
            # Force redistribute stations from oversized clusters to undersized ones
            for cid, size in undersized_clusters:
                needed_stations = min_bikes - size
                oversized_clusters = [(ocid, osize) for ocid, osize in cluster_sizes.items() 
                                    if osize > target_cluster_size]
                
                if oversized_clusters:
                    for ocid, _ in oversized_clusters:
                        cluster_stations = station_data[station_data['cluster'] == ocid]
                        for _, station in cluster_stations.iterrows():
                            if cluster_sizes[cid] >= min_bikes:
                                break
                            station_data.loc[station.name, 'cluster'] = cid
                            cluster_sizes[cid] = cluster_sizes.get(cid, 0) + 1
                            cluster_sizes[ocid] = cluster_sizes[ocid] - 1
        
        # Then handle flow violations
        clusters_by_violation = []
        for cluster_id in range(optimal_clusters):
            if cluster_id not in cluster_sizes:
                continue
                
            flow = cluster_flows.get(cluster_id, 0)
            size = cluster_sizes.get(cluster_id, 0)
            
            flow_violation = max(0, abs(flow) - max_flow)
            size_violation = max(0, size - max_bikes, min_bikes - size) * 100
            
            total_violation = flow_violation * 2 + size_violation
            if total_violation > 0:
                clusters_by_violation.append((cluster_id, total_violation, flow_violation))
        
        clusters_by_violation.sort(key=lambda x: (x[2], x[1]), reverse=True)
        
        if not clusters_by_violation:
            print("All clusters balanced!")
            break
            
        for cluster_id, violation, flow_violation in clusters_by_violation:
            flow = cluster_flows[cluster_id]
            size = cluster_sizes[cluster_id]
            
            print(f"Balancing cluster {cluster_id} (flow: {flow:.1f}, size: {size})")
            attempts = 0
            
            while attempts < MAX_ATTEMPTS_PER_CLUSTER:
                cluster_stations = station_data[station_data['cluster'] == cluster_id]
                
                # Prioritize moving stations based on which constraint is violated more
                if size > max_bikes:
                    # If oversized, sort by absolute flow contribution to minimize impact
                    stations_to_move = cluster_stations.sort_values('net_flow', key=abs)
                else:
                    # If flow violated, sort by flow contribution
                    stations_to_move = cluster_stations.sort_values('net_flow', 
                        ascending=(flow > max_flow))
                
                moved = False
                for _, station in stations_to_move.iterrows():
                    best_cluster = find_best_cluster_for_station(
                        station, cluster_id, cluster_flows, cluster_sizes, 
                        strict=(attempts < MAX_ATTEMPTS_PER_CLUSTER // 2)
                    )
                    
                    if best_cluster is not None:
                        station_data.loc[station.name, 'cluster'] = best_cluster
                        # Update metrics
                        cluster_flows[cluster_id] -= station['net_flow']
                        cluster_flows[best_cluster] = cluster_flows.get(best_cluster, 0) + station['net_flow']
                        cluster_sizes[cluster_id] -= 1
                        cluster_sizes[best_cluster] = cluster_sizes.get(best_cluster, 0) + 1
                        moved = True
                        print(f"  Moved station (flow={station['net_flow']:.1f}) to cluster {best_cluster}")
                        break
                
                if not moved:
                    attempts += 1
                
                current_flow = cluster_flows[cluster_id]
                current_size = cluster_sizes[cluster_id]
                
                if (min_flow <= current_flow <= max_flow and 
                    min_bikes <= current_size <= max_bikes):
                    print(f"  Cluster {cluster_id} balanced: Flow={current_flow:.1f}, Size={current_size}")
                    break
                    
                if attempts == MAX_ATTEMPTS_PER_CLUSTER:
                    print(f"  Warning: Max attempts reached for cluster {cluster_id}")

    return station_data

def compute_violation(clustered_data: pd.DataFrame, min_flow: int, max_flow: int) -> float:
    """
    Compute the total flow violation across all clusters.
    Returns both the total violation and the number of violating clusters.
    """
    total_violation = 0
    violating_clusters = 0
    for cluster_id, group in clustered_data.groupby('cluster'):
        net_flow = group['net_flow'].sum()
        if net_flow > max_flow:
            total_violation += net_flow - max_flow
            violating_clusters += 1
        elif net_flow < min_flow:
            total_violation += min_flow - net_flow
            violating_clusters += 1
    return total_violation, violating_clusters

def run_cluster_instance(args: Tuple[pd.DataFrame, List[str], int]) -> Tuple[int, pd.DataFrame, float, int]:
    """
    Run a single clustering instance with the provided random seed.
    Returns (seed, resulting DataFrame, total flow violation, number of violating clusters).
    """
    station_features, manual_stations, random_seed = args
    result = dynamic_kmeans(
        station_features.copy(),
        manual_stations,
        min_flow=-80,
        max_flow=80,
        min_bikes=15,
        max_bikes=35,
        min_clusters=10,
        max_clusters=50,
        random_seed=random_seed
    )
    violation, violating_clusters = compute_violation(result, -80, 80)
    return (random_seed, result, violation, violating_clusters)

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
        if all((-100 <= flow <= 100) for flow in period_totals):
            valid_clusters.append(cluster)
    
    return valid_clusters

def print_cluster_statistics(clustered_data: pd.DataFrame):
    """Print detailed statistics for each cluster"""
    clusters = clustered_data['cluster'].unique()
    clusters.sort()
    
    print("\nFinal cluster statistics:")
    for cluster_id in clusters:
        cluster_stations = clustered_data[clustered_data['cluster'] == cluster_id]
        size = len(cluster_stations)
        flow = cluster_stations['net_flow'].sum()
        print(f"Cluster {cluster_id}: Size={size}, Flow={flow:.1f}")
        
        # Print sample stations (up to 5)
        sample_stations = cluster_stations['station_name'].sample(min(5, size)).tolist()
        print(f"Sample stations: {sample_stations}\n")

def main():
    # Load data
    coords_data = load_station_coordinates()
    
    try: 
        file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            "Bike share ridership 2023-07.csv"
        )
        # Example: July, Tuesday, AM
        flow_data = prepare_flow_data(
            file_path,
            month=7,
            day_of_week='Tuesday',
            period='AM'
        )
        print(f"Successfully loaded bike data")
    except FileNotFoundError:
        print(f"Warning: Data file not found: {file_path}")
        raise
    
    # Create features
    station_features, manual_stations = create_station_features(coords_data, flow_data)
    
    # Get manual clusters
    manual_clusters = get_manual_clusters()
    
    # Run multiple instances with different random seeds
    NUM_RUNS = 7
    # Use different random seeds for each run
    random_seeds = [42, 123, 456, 789, 101112, 131415, 161718]  # Different seeds for variety
    
    print(f"\nRunning {NUM_RUNS} clustering instances with different seeds...")
    
    best_result = None
    best_violation = float('inf')
    best_violating_clusters = float('inf')
    
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(run_cluster_instance, (station_features.copy(), manual_stations, seed))
            for seed in random_seeds
        ]
        
        for future in as_completed(futures):
            seed, result, violation, violating_clusters = future.result()
            print(f"\nRun with seed {seed} results:")
            print(f"Total flow violation: {violation:.1f}")
            print(f"Number of violating clusters: {violating_clusters}")
            
            # Prefer solutions with fewer violating clusters, then lower total violation
            if (violating_clusters < best_violating_clusters or 
                (violating_clusters == best_violating_clusters and violation < best_violation)):
                best_result = result
                best_violation = violation
                best_violating_clusters = violating_clusters
                print(f"New best solution found with seed {seed}!")
    
    print(f"\nSelected best solution with {best_violating_clusters} violating clusters")
    print(f"Total flow violation: {best_violation:.1f}")
    
    # Convert the best result DataFrame to cluster format
    final_clusters = []
    for cluster_id in best_result['cluster'].unique():
        cluster_stations = best_result[best_result['cluster'] == cluster_id]
        station_names = cluster_stations['station_name'].tolist()
        final_clusters.append(station_names)
    
    # Add manual clusters to the beginning
    all_clusters = []
    for cluster_name, stations in manual_clusters.items():
        all_clusters.append(stations)
    all_clusters.extend(final_clusters)
    
    # Save results
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, "station_clusters.json")
    
    with open(output_path, "w") as f:
        json.dump({
            "clusters": all_clusters,
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"\nCreated {len(all_clusters)} clusters")
    print(f"Total stations clustered: {sum(len(c) for c in all_clusters)}")
    print(f"Clusters saved to: {output_path}")
    
    # Print detailed cluster information
    print_cluster_statistics(best_result)

if __name__ == "__main__":
    main()
