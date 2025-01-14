from flask import Flask
from apscheduler.schedulers.background import BackgroundScheduler

# Import the necessary functions and data
from bike_counts import fetch_bike_counts, check_for_rebalancing
from calculate_clusters import execute_queries_on_remote, stations_info  # Adjust import as needed

app = Flask(__name__)

def update_bike_counts_and_check_rebalancing():
    # Fetch updated bike counts
    bike_counts = fetch_bike_counts()

    # Update stations_info with new bike counts
    for station_info in stations_info:
        station_id = station_info['station_id']
        station_info['bike_count'] = bike_counts.get(station_id, 0)

    # Check for rebalancing
    check_for_rebalancing(stations_info)

    # Optionally update the database with new bike counts
    update_queries = []
    for station_info in stations_info:
        update_queries.append({
            "query": """
                MATCH (s:Station {name: $name})
                SET s.bike_count = $bike_count
            """,
            "parameters": {
                "name": station_info['name'],
                "bike_count": station_info['bike_count']
            }
        })
    # Execute the update queries
    remote_uri = "bolt://localhost:7687"  # Replace with your Neo4j URI
    username = "neo4j"  # Replace with your username
    password = "testpassword"  # Replace with your password
    execute_queries_on_remote(update_queries, remote_uri, username, password)

# Initialize the scheduler
scheduler = BackgroundScheduler()
scheduler.add_job(func=update_bike_counts_and_check_rebalancing, trigger="interval", minutes=5)
scheduler.start()

# Shut down the scheduler when exiting the app
import atexit
atexit.register(lambda: scheduler.shutdown())

if __name__ == "__main__":
    app.run()
