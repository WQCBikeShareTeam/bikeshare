import json
import os

def view_current_clusters():
    """Display the current clusters from the JSON file"""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        clusters_path = os.path.join(current_dir, "station_clusters.json")
        
        with open(clusters_path, 'r') as f:
            data = json.load(f)
            
        print(f"\nCluster Summary (as of {data['timestamp']}):")
        print("=" * 50)
        
        for i, cluster in enumerate(data['clusters']):
            print(f"\nCluster {i}:")
            print(f"Number of stations: {len(cluster)}")
            print("Stations:")
            for station in cluster:
                print(f"  - {station}")
            print("-" * 30)
            
        print(f"\nTotal Clusters: {len(data['clusters'])}")
        print(f"Total Stations: {sum(len(c) for c in data['clusters'])}")
        
    except Exception as e:
        print(f"Error reading clusters: {e}")

if __name__ == "__main__":
    view_current_clusters() 