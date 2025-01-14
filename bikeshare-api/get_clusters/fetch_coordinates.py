import os
import json
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
API_KEY = os.getenv("OPENCAGE_API_KEY")

CACHE_FILE = "station_coordinates_cache.json"

# Load cache if it exists
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r") as f:
        coordinates_cache = json.load(f)
else:
    coordinates_cache = {}

def get_coordinates(station_name, city="Toronto"):
    """
    Fetches latitude and longitude for a station from cache or API.
    """
    # Check cache first
    if station_name in coordinates_cache:
        return coordinates_cache[station_name]

    # Fetch from OpenCage API if not in cache
    url = "https://api.opencagedata.com/geocode/v1/json"
    params = {"q": f"{station_name}, {city}", "key": API_KEY}
    response = requests.get(url, params=params)
    data = response.json()

    if response.status_code == 200 and data["results"]:
        coords = data["results"][0]["geometry"]
        coordinates_cache[station_name] = {"latitude": coords["lat"], "longitude": coords["lng"]}

        # Save updated cache
        with open(CACHE_FILE, "w") as f:
            json.dump(coordinates_cache, f, indent=4)

        return coordinates_cache[station_name]
    else:
        print(f"Error: Unable to fetch coordinates for {station_name}")
        return None
