from fetch_coordinates import get_coordinates
from math import radians, cos, sin, sqrt, atan2
from neo4j import GraphDatabase # type: ignore

# Load all station names from a text file
with open("unique_station_names.txt", "r") as f:
    all_stations = [line.strip() for line in f.readlines()]

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points on the Earth (Haversine formula).
    """
    R = 6371  # Earth radius in kilometers
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def generate_neo4j_queries(stations_info):
    """
    Generate Neo4j Cypher queries to add stations and relationships between all pairs.
    """
    queries = []

    # Create nodes for all stations
    for station_info in stations_info:
        station = station_info['name']
        coords = station_info['coords']

        queries.append({
            "query": """
                MERGE (:Station {name: $name, latitude: $latitude, longitude: $longitude})
            """,
            "parameters": {
                "name": station,
                "latitude": coords['latitude'],
                "longitude": coords['longitude']
            }
        })

    # Create relationships between all pairs of stations
    for i in range(len(stations_info)):
        for j in range(i + 1, len(stations_info)):
            station_info_a = stations_info[i]
            station_info_b = stations_info[j]

            station_a = station_info_a['name']
            coords_a = station_info_a['coords']
            station_b = station_info_b['name']
            coords_b = station_info_b['coords']

            distance = haversine(
                coords_a['latitude'], coords_a['longitude'],
                coords_b['latitude'], coords_b['longitude']
            )

            # Add the relationship in one direction
            queries.append({
                "query": """
                    MATCH (a:Station {name: $station_a}),
                          (b:Station {name: $station_b})
                    MERGE (a)-[:CONNECTED_TO {distance: $distance}]->(b)
                """,
                "parameters": {
                    "station_a": station_a,
                    "station_b": station_b,
                    "distance": round(distance, 2)
                }
            })

    return queries

def find_nearby_stations(center_station, all_stations, radius_km=None, max_stations=None):
    """
    Find nearby stations based on distance from the center station.
    Include the center station in the returned list.
    """
    center_coords = get_coordinates(center_station)
    if not center_coords:
        print(f"Error: Unable to fetch coordinates for center station: {center_station}")
        return []

    stations_info = []

    # Include the center station
    stations_info.append({
        'name': center_station,
        'coords': center_coords
    })

    for station in all_stations:
        if station == center_station:
            continue  # Skip the center station (already added)

        coords = get_coordinates(station)
        if not coords:
            print(f"Error: Unable to fetch coordinates for station: {station}")
            continue

        distance = haversine(
            center_coords['latitude'], center_coords['longitude'],
            coords['latitude'], coords['longitude']
        )

        if radius_km is not None and distance > radius_km:
            continue  # Skip stations beyond the specified radius

        stations_info.append({
            'name': station,
            'coords': coords,
            'distance': distance  # Store distance for sorting
        })

    # If max_stations is specified, sort and limit the number of stations
    if max_stations is not None:
        # Sort stations by distance from center
        stations_info = stations_info[:1] + sorted(stations_info[1:], key=lambda x: x['distance'])[:max_stations]

    return stations_info

def execute_queries_on_remote(queries, remote_uri, username, password):
    """
    Execute generated queries on the remote Neo4j instance.
    """
    driver = GraphDatabase.driver(remote_uri, auth=(username, password))
    try:
        with driver.session() as session:
            for item in queries:
                try:
                    session.write_transaction(lambda tx: tx.run(item["query"], **item["parameters"]))
                    print(f"Successfully executed query for {item['parameters'].get('name', 'relationship')}")
                except Exception as e:
                    print(f"Failed to execute query: {item['query']}\nError: {e}")
    finally:
        driver.close()

if __name__ == "__main__":
    # Prompt the user for the center station
    center_station = input("Enter the name of the center station: ").strip()

    # Choose method: specify radius or number of closest stations
    method = input("Do you want to specify a radius (r) or number of closest stations (n)? [r/n]: ").strip().lower()

    if method == 'r':
        # User specifies a radius in kilometers
        radius_km = float(input("Enter the radius in kilometers: "))
        stations_info = find_nearby_stations(center_station, all_stations, radius_km=radius_km)
    elif method == 'n':
        # User specifies the number of closest stations
        max_stations = int(input("Enter the number of closest stations to find: "))
        stations_info = find_nearby_stations(center_station, all_stations, max_stations=max_stations)
    else:
        print("Invalid option selected. Exiting.")
        exit()

    # Generate the Neo4j queries
    queries = generate_neo4j_queries(stations_info)

    # Output queries to a file
    output_file = "neo4j_queries.cypher"
    with open(output_file, "w") as f:
        for item in queries:
            f.write(item["query"] + "\n")

    print(f"Generated {len(queries)} Cypher queries and saved to {output_file}.")

    # Execute queries on a remote Neo4j instance
    remote_uri = "bolt://localhost:7687"  # Replace with your Neo4j URI
    username = "neo4j"  # Replace with your username
    password = "testpassword"  # Replace with your password

    execute_queries_on_remote(queries, remote_uri, username, password)
