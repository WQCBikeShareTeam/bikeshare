from neo4j import GraphDatabase # type: ignore
import requests # type: ignore
from math import radians, cos, sin, sqrt, atan2

def get_station_info_from_db(remote_uri, username, password):
    """
    Connects to Neo4j database and retrieves all station names and coordinates.
    """
    driver = GraphDatabase.driver(remote_uri, auth=(username, password))
    stations = []
    try:
        with driver.session() as session:
            result = session.run("MATCH (s:Station) RETURN s.name AS name, s.latitude AS latitude, s.longitude AS longitude")
            for record in result:
                stations.append({
                    "name": record["name"],
                    "latitude": record["latitude"],
                    "longitude": record["longitude"]
                })
    finally:
        driver.close()
    return stations

def fetch_station_info():
    """
    Fetches station information from Bike Share Toronto's GBFS feed.
    Returns a list of stations with their names, IDs, and coordinates.
    """
    url = "https://tor.publicbikesystem.net/ube/gbfs/v1/en/station_information"
    response = requests.get(url)
    if response.status_code != 200:
        print("Error fetching station information.")
        return []
    data = response.json()
    station_info = []
    for station in data['data']['stations']:
        station_info.append({
            "name": station['name'],
            "station_id": station['station_id'],
            "latitude": station['lat'],
            "longitude": station['lon']
        })
    return station_info

def fetch_bike_counts():
    """
    Fetches real-time bike counts from Bike Share Toronto's GBFS feed.
    Returns a dictionary mapping station IDs to bike counts.
    """
    url = "https://tor.publicbikesystem.net/ube/gbfs/v1/en/station_status"
    response = requests.get(url)
    if response.status_code != 200:
        print("Error fetching bike counts.")
        return {}
    data = response.json()
    bike_counts = {}
    for station in data['data']['stations']:
        station_id = station['station_id']
        num_bikes = station['num_bikes_available']
        bike_counts[station_id] = num_bikes
    return bike_counts

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points on the Earth (Haversine formula).
    """
    R = 6371  # Earth radius in kilometers
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def match_station_by_coordinates(station, gbfs_stations, max_distance=0.05):
    """
    Attempts to match a station to a GBFS station based on coordinates.
    Returns the station_id if a match is found within the max_distance (in km).
    """
    lat1 = station['latitude']
    lon1 = station['longitude']
    for gbfs_station in gbfs_stations:
        lat2 = gbfs_station['latitude']
        lon2 = gbfs_station['longitude']
        distance = haversine(lat1, lon1, lat2, lon2)
        if distance <= max_distance:
            return gbfs_station['station_id']
    return None

def check_for_rebalancing(station_bike_counts, threshold_low=5, threshold_high=15):
    """
    Checks the bike counts and writes to a text file if rebalancing is needed.
    """
    with open('rebalance_log.txt', 'a') as f:
        for station_name, bike_count in station_bike_counts.items():
            if bike_count < threshold_low or bike_count > threshold_high:
                f.write(f"Rebalance needed at station: {station_name} (Bike count: {bike_count})\n")

def main():
    # Replace with your Neo4j credentials
    remote_uri = "neo4j+s://37f0488d.databases.neo4j.io"  # Replace with your Neo4j URI
    username = "neo4j"  # Replace with your username
    password = "Q0YsXHrcDzwbArLutk837JcuRzuG_jBLQo3QaogSE_U"  # Replace with your password

    # Get stations from the database
    stations = get_station_info_from_db(remote_uri, username, password)
    if not stations:
        print("No stations found in the database.")
        return

    # Get GBFS station information
    gbfs_stations = fetch_station_info()
    if not gbfs_stations:
        print("Error fetching station information.")
        return

    # Fetch bike counts
    station_id_to_bike_count = fetch_bike_counts()
    if not station_id_to_bike_count:
        print("Error fetching bike counts.")
        return

    # Create a mapping of station names to bike counts
    station_bike_counts = {}
    for station in stations:
        station_name = station['name']
        # Try to find station ID by exact name match
        station_id = next((s['station_id'] for s in gbfs_stations if s['name'] == station_name), None)
        if station_id is None:
            # Try to match by coordinates
            station_id = match_station_by_coordinates(station, gbfs_stations)
            if station_id is None:
                print(f"Station ID not found for station: {station_name}")
                continue
        bike_count = station_id_to_bike_count.get(station_id, 0)
        station_bike_counts[station_name] = bike_count

    # Check for rebalancing and write to text file
    check_for_rebalancing(station_bike_counts)

if __name__ == "__main__":
    main()
