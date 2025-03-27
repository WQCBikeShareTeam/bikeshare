"""
Refactored Bikeshare Demand and Optimization Pipeline
"""

import os
import json
import time
import copy
import traceback
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ============================
# Configuration Constants
# ============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "cache")
EXPORT_DIR = os.path.join(BASE_DIR, "exports")

# API keys and file paths (update these as needed)
OPENWEATHER_API_KEY = "46574182efce52561d5b815bf2c3c5d2"
# GOOGLE_API_KEY can be set if using external route APIs
GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY"

# Default operational parameters
DEFAULT_STARTING_BIKES = 4
STATION_CAPACITY = 25
TRUCK_CAPACITY = 80

# Cost parameters for travel costs
COST_PER_KM = 0.50       # $ per kilometer
COST_PER_HOUR = 10.00    # $ per hour (driver cost)
WEAR_TEAR_PER_KM = 0.15  # Maintenance cost per kilometer

# Cost scale ( how much cost affects selections )
COST_SCALE = 0.5

# ============================
# Utility Functions
# ============================
def calculate_distance(coord1, coord2):
    """
    Approximate the distance (in km) between two (lat, lon) points using a simple Euclidean formula.
    (1° latitude ≈ 111 km; 1° longitude ≈ 81 km at Toronto's latitude)
    """
    lat_diff = (coord1[0] - coord2[0]) * 111
    lon_diff = (coord1[1] - coord2[1]) * 81
    return (lat_diff**2 + lon_diff**2)**0.5

def calculate_cluster_center(stations):
    """Calculate the center (average latitude and longitude) of a list of stations."""
    if not stations:
        return (0, 0)
    avg_lat = sum(s['latitude'] for s in stations) / len(stations)
    avg_lon = sum(s['longitude'] for s in stations) / len(stations)
    return (avg_lat, avg_lon)

# ============================
# Data Loading Functions
# ============================
def load_stations_data():
    """
    Load station data (coordinates, names, etc.) from a JSON file.
    """
    try:
        path = os.path.join(BASE_DIR, "..", "bikeshare-api", "get_clusters", "station_coordinates_cache.json")
        with open(path, 'r') as f:
            data = json.load(f)
        stations = {}
        station_id = 7000
        for name, coords in data.items():
            stations[name] = {
                'station_id': str(station_id),
                'name': name,
                'latitude': coords['latitude'],
                'longitude': coords['longitude']
            }
            station_id += 1
        return stations
    except Exception as e:
        print(f"Error loading station data: {e}")
        return {}

def load_historical_data():
    """
    Load historical trip data from CSV.
    (Update the file path as needed.)
    """
    try:
        data_file = os.path.join(BASE_DIR, "data", "Bike_share_ridership.csv")
        df = pd.read_csv(data_file, encoding='utf-8', errors='replace')
        df['Start Time'] = pd.to_datetime(df['Start Time'])
        df['End Time'] = pd.to_datetime(df['End Time'])
        return df
    except Exception as e:
        print(f"Error loading historical data: {e}")
        raise

def load_clusters_with_coordinates():
    """
    Load clusters from JSON and enrich them with station coordinates.
    (Update file paths as needed.)
    """
    try:
        clusters_path = os.path.join(BASE_DIR, "..", "Region_Creation_Parsing", "Region_Creation_Parsing", "station_clusters.json")
        coords_path = os.path.join(BASE_DIR, "..", "bikeshare-api", "get_clusters", "station_coordinates_cache.json")
        with open(clusters_path, 'r') as f:
            clusters_data = json.load(f)
        with open(coords_path, 'r') as f:
            coords_data = json.load(f)
        cleaned_coords = {name.strip().lower(): coords for name, coords in coords_data.items()}
        enriched_clusters = {}
        for cluster_id, station_names in enumerate(clusters_data['clusters']):
            station_list = []
            for name in station_names:
                key = name.strip().lower()
                if key in cleaned_coords:
                    station_list.append({
                        'name': name,
                        'latitude': cleaned_coords[key]['latitude'],
                        'longitude': cleaned_coords[key]['longitude'],
                        'predictions': {}
                    })
                else:
                    print(f"Warning: No coordinates found for station: {name}")
            if station_list:
                enriched_clusters[cluster_id] = station_list
        return enriched_clusters
    except Exception as e:
        print(f"Error loading clusters: {e}")
        raise

# ============================
# Weather Functions
# ============================
class Precipitation(Enum):
    NONE = "none"
    LIGHT_RAIN = "light rain"
    HEAVY_RAIN = "heavy rain"
    LIGHT_SNOW = "light snow"
    HEAVY_SNOW = "heavy snow"

@dataclass
class WeatherZoneLocation:
    zone: str
    latitude: float
    longitude: float
    filepath: str

# Define weather zone locations (update file paths as needed)
WEATHER_ZONE_LOCATIONS = [
    WeatherZoneLocation("Toronto City Center", 43.63, -79.4, os.path.join(BASE_DIR, "data", "Toronto_City_Center_Weather.csv")),
    WeatherZoneLocation("Toronto City", 43.67, -79.40, os.path.join(BASE_DIR, "data", "Toronto_Weather.csv")),
    WeatherZoneLocation("Toronto Intl Airport", 43.68, -79.63, os.path.join(BASE_DIR, "data", "TorontoAirportWeather.csv"))
]

class WeatherService:
    def __init__(self, api_key=OPENWEATHER_API_KEY):
        self.api_key = api_key
        self.base_url = "http://api.openweathermap.org/data/2.5/forecast"
        self.cache = {}
        self.cache_duration = timedelta(minutes=30)
        self.last_api_call = 0
        self.rate_limit_delay = 1.0

    def _respect_rate_limit(self):
        current = time.time()
        if current - self.last_api_call < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - (current - self.last_api_call))
        self.last_api_call = time.time()

    def get_weather_forecast(self, lat, lon):
        cache_key = f"{lat:.4f},{lon:.4f}"
        now = datetime.now()
        if cache_key in self.cache:
            data, timestamp = self.cache[cache_key]
            if now - timestamp < self.cache_duration:
                return data
        self._respect_rate_limit()
        params = {"lat": lat, "lon": lon, "appid": self.api_key, "units": "metric", "cnt": 8}
        try:
            response = requests.get(self.base_url, params=params)
            if response.status_code == 200:
                data = response.json()
                forecast = self._process_forecast(data)
                self.cache[cache_key] = (forecast, now)
                return forecast
            else:
                print(f"Weather API error: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error in weather API call: {e}")
            return None

    def _process_forecast(self, data):
        forecasts = []
        for item in data.get('list', []):
            temp = item['main']['temp']
            pop = item.get('pop', 0)
            precip = Precipitation.NONE.value
            if pop >= 0.3:
                if temp > 0:
                    rain = item.get('rain', {}).get('3h', 0) / 3
                    if rain > 7.5:
                        precip = Precipitation.HEAVY_RAIN.value
                    elif rain > 0:
                        precip = Precipitation.LIGHT_RAIN.value
                else:
                    snow = item.get('snow', {}).get('3h', 0) / 3
                    if snow > 4.0:
                        precip = Precipitation.HEAVY_SNOW.value
                    elif snow > 0:
                        precip = Precipitation.LIGHT_SNOW.value
            forecasts.append({
                'temperature': categorize_temperature(temp),
                'precipitation': precip,
                'raw_temp': temp,
                'timestamp': item['dt']
            })
        return forecasts

def categorize_temperature(temp):
    """Convert a numeric temperature into a category string."""
    if temp < -10:
        return "below -10°C"
    elif temp < 0:
        return "-5°C to 0°C"
    elif temp < 5:
        return "0°C to 5°C"
    elif temp < 10:
        return "5°C to 10°C"
    elif temp < 15:
        return "10°C to 15°C"
    elif temp < 20:
        return "15°C to 20°C"
    elif temp < 25:
        return "20°C to 25°C"
    else:
        return "above 25°C"

def load_historical_weather_data():
    """Load historical weather data from CSV files for each weather zone."""
    weather_data = {}
    for zone_loc in WEATHER_ZONE_LOCATIONS:
        try:
            if not os.path.exists(zone_loc.filepath):
                print(f"File not found: {zone_loc.filepath}")
                continue
            df = pd.read_csv(zone_loc.filepath)
            df.columns = [col.strip().strip('"').strip("'") for col in df.columns]
            df['datetime'] = pd.to_datetime(df['Time (LST)'])
            df['is_rain'] = df['Weather'].fillna('').str.contains('rain', case=False)
            df['is_snow'] = df['Weather'].fillna('').str.contains('snow', case=False)
            df['temperature'] = df['Temp (°C)']
            df['precipitation'] = df['Precip. Amount (mm)']
            weather_data[zone_loc.zone] = df
            print(f"Loaded weather data for {zone_loc.zone} ({len(df)} records)")
        except Exception as e:
            print(f"Error loading weather data for {zone_loc.zone}: {e}")
    return weather_data

def assign_stations_to_zones(stations_data):
    """Assign each station to the nearest weather zone."""
    for station in stations_data.values():
        coords = (station['latitude'], station['longitude'])
        closest_zone = None
        min_dist = float('inf')
        for zone_loc in WEATHER_ZONE_LOCATIONS:
            dist = calculate_distance(coords, (zone_loc.latitude, zone_loc.longitude))
            if dist < min_dist:
                min_dist = dist
                closest_zone = zone_loc.zone
        station['weather_zone'] = closest_zone
    return stations_data

def merge_weather_with_historical_data(historical_data, stations_data, weather_data):
    """
    Merge historical trip data with weather data based on station zones and time.
    (This function uses simplified matching on month, day, and hour.)
    """
    result = historical_data.copy()
    if 'Start Time' in result.columns:
        result['Start Time'] = pd.to_datetime(result['Start Time'])
        result['hour'] = result['Start Time'].dt.hour
        result['day'] = result['Start Time'].dt.day
        result['month'] = result['Start Time'].dt.month
    else:
        return result

    if 'temperature' not in result.columns:
        result['temperature'] = None
    if 'precipitation' not in result.columns:
        result['precipitation'] = "none"

    for idx, row in result.iterrows():
        station_name = row.get('Start Station Name')
        if not station_name or station_name not in stations_data:
            continue
        zone = stations_data[station_name].get('weather_zone')
        if not zone or zone not in weather_data:
            continue
        zone_df = weather_data[zone]
        month, day, hour = row['month'], row['day'], row['hour']
        matches = zone_df[(zone_df['Month'] == month) & (zone_df['Day'] == day)]
        if not matches.empty:
            weather_record = matches.iloc[0]
            result.at[idx, 'temperature'] = weather_record.get('Temp (°C)', None)
            result.at[idx, 'precipitation'] = weather_record.get('Precip. Amount (mm)', 0)
    return result

# ============================
# Prediction Model Functions
# ============================
def preprocess_and_cache_data(force_refresh=False):
    """
    Preprocess historical data (merge with weather) and cache the result to a CSV.
    Returns the cache file path.
    """
    cache_file = os.path.join(CACHE_DIR, "merged_historical_weather_data.csv")
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    if os.path.exists(cache_file) and not force_refresh:
        try:
            pd.read_csv(cache_file, nrows=5)
            return cache_file
        except Exception:
            force_refresh = True
    historical_data = load_historical_data()
    weather_data = load_historical_weather_data()
    stations_data = load_stations_data()
    stations_data = assign_stations_to_zones(stations_data)
    merged = merge_weather_with_historical_data(historical_data, stations_data, weather_data)
    merged.to_csv(cache_file, index=False)
    return cache_file

def train_prediction_model(station_name: str) -> (dict, dict):
    """
    Train a prediction model (RandomForest) for a given station.
    (For simplicity, only the net flow is modeled.)
    """
    try:
        cache_file = preprocess_and_cache_data(force_refresh=False)
        data = pd.read_csv(cache_file)
        data['Start Time'] = pd.to_datetime(data['Start Time'])
        data = data[data['Start Time'].dt.month == 3]  # Example: use March data
    except Exception as e:
        print(f"Error loading cached data: {e}")
        data = load_historical_data()

    data['DayOfWeek'] = data['Start Time'].dt.day_name()
    # Filter data for the station
    outflow = data[data['Start Station Name'] == station_name].copy()
    inflow = data[data['End Station Name'] == station_name].copy()
    if not outflow.empty:
        outflow['flow'] = -1
    if not inflow.empty:
        inflow['flow'] = 1
    flows = pd.concat([outflow, inflow])
    flows['Hour'] = flows['Start Time'].dt.hour
    group_cols = ['Hour', 'DayOfWeek']
    agg_dict = {'flow': 'sum'}
    station_data = flows.groupby(group_cols, dropna=False).agg(agg_dict).reset_index()

    # Encode day of week
    le_day = LabelEncoder()
    le_day.fit(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    station_data['DayOfWeek_encoded'] = le_day.transform(station_data['DayOfWeek'])

    # For simplicity, use fixed dummy values for temperature/precipitation encoding
    station_data['temp_encoded'] = 3
    station_data['precip_encoded'] = 0

    features = station_data[['Hour', 'DayOfWeek_encoded', 'temp_encoded', 'precip_encoded']]
    target = station_data['flow']

    if target.nunique() > 1:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"Model for {station_name}: MSE={mse:.2f}, R²={r2:.2f}")
    else:
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(features, target)

    models = {'flow': model}
    encoders = {'day': le_day, 'temp': None, 'precip': None}
    return models, encoders

def get_station_predictions(station: dict, start_time: datetime) -> list:
    """
    Generate 24-hour predictions for a station using the trained model.
    """
    models, encoders = train_prediction_model(station['name'])
    predictions = []
    current_bikes = DEFAULT_STARTING_BIKES
    weather_service = WeatherService()
    wf = weather_service.get_weather_forecast(station['latitude'], station['longitude'])
    
    if wf:
        if len(wf) < 24:
            # Pad the forecast by repeating the last available forecast
            last_entry = wf[-1] if wf else {'temperature': "15°C to 20°C", 'precipitation': "none"}
            forecast = wf + [last_entry] * (24 - len(wf))
        else:
            forecast = wf[:24]
    else:
        forecast = [{'temperature': "15°C to 20°C", 'precipitation': "none"}] * 24

    for i in range(24):
        prediction_time = start_time + timedelta(hours=i)
        hour = prediction_time.hour
        day_name = prediction_time.strftime('%A')
        day_encoded = encoders['day'].transform([day_name])[0]
        features = {
            'Hour': hour,
            'DayOfWeek_encoded': day_encoded,
            'temp_encoded': 3,    # Dummy value
            'precip_encoded': 0   # Dummy value
        }
        features_df = pd.DataFrame([features])
        pred_flow = models['flow'].predict(features_df)[0]
        current_bikes = max(0, min(STATION_CAPACITY, current_bikes + pred_flow))
        predictions.append({
            'hour': prediction_time.strftime('%Y-%m-%d %H:00:00'),
            'net_flow': round(pred_flow, 2),
            'predicted_bikes': round(current_bikes),
            'temperature': forecast[i]['temperature'],
            'precipitation': forecast[i]['precipitation']
        })
    return predictions

def calculate_station_payout(predictions: list, station_capacity: int = STATION_CAPACITY) -> list:
    """
    Calculate payout metrics (dummy calculation) based on predicted net flows.
    """
    for pred in predictions:
        # In this dummy example, loss is proportional to the absolute net flow.
        pred['hourly_loss'] = abs(pred['net_flow']) * 1.0
    return predictions

def print_station_predictions(predictions: list):
    if not predictions:
        print("No predictions available.")
        return
    header = f"{'Hour':<20} {'Net Flow':>10} {'Predicted Bikes':>15} {'Temp':>15} {'Precip':>15} {'Loss':>10}"
    print("\n24-Hour Station Predictions:")
    print(header)
    print("-" * len(header))
    for pred in predictions:
        row = f"{pred['hour']:<20} {pred['net_flow']:>10.2f} {pred['predicted_bikes']:>15} {pred['temperature']:>15} {pred['precipitation']:>15} {pred['hourly_loss']:>10.2f}"
        print(row)

# ============================
# Clustering and Optimization Functions
# ============================
def load_predictions_from_csv(filename="hourly_predictions.csv") -> dict:
    """
    Load station predictions from a CSV file.
    """
    csv_path = os.path.join(EXPORT_DIR, filename)
    if not os.path.exists(csv_path):
        print(f"CSV file not found at {csv_path}")
        return {}
    station_predictions = {}
    try:
        import csv
        with open(csv_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                name = row['station_name']
                if name not in station_predictions:
                    station_predictions[name] = []
                for field in ['net_flow', 'predicted_bikes', 'hourly_loss']:
                    try:
                        row[field] = float(row[field])
                    except:
                        pass
                station_predictions[name].append(row)
        for name in station_predictions:
            station_predictions[name] = sorted(station_predictions[name], key=lambda x: x['hour'])
        return station_predictions
    except Exception as e:
        print(f"Error loading predictions from CSV: {e}")
        return {}

def export_predictions_to_csv(clusters, cluster_id, filename="hourly_predictions.csv"):
    """
    Export hourly predictions for each station in a cluster to CSV.
    """
    if cluster_id not in clusters:
        print(f"Cluster {cluster_id} not found for CSV export.")
        return
    if not os.path.exists(EXPORT_DIR):
        os.makedirs(EXPORT_DIR, exist_ok=True)
    output_file = os.path.join(EXPORT_DIR, filename)
    rows = []
    for station in clusters[cluster_id]:
        name = station['name']
        if 'predictions' in station and station['predictions']:
            for pred in station['predictions']:
                row = {
                    'station_name': name,
                    'station_latitude': station['latitude'],
                    'station_longitude': station['longitude'],
                    'hour': pred.get('hour', ''),
                    'net_flow': pred.get('net_flow', 0),
                    'predicted_bikes': pred.get('predicted_bikes', 0),
                    'temperature': pred.get('temperature', ''),
                    'precipitation': pred.get('precipitation', ''),
                    'hourly_loss': pred.get('hourly_loss', 0)
                }
                rows.append(row)
    if rows:
        import csv
        with open(output_file, 'w', newline='') as csvfile:
            fieldnames = list(rows[0].keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Predictions exported to {output_file}")
    else:
        print("No prediction data available to export.")

def create_reduced_clusters(cluster_id=4, max_distance=1, clusters=None, use_cached_predictions=True):
    """
    Create reduced clusters by grouping stations needing bike adjustments.
    (For simplicity, this example only groups stations that already show a nonzero optimal adjustment.)
    """
    if clusters is None:
        clusters = load_clusters_with_coordinates()
    if cluster_id not in clusters:
        print(f"Cluster {cluster_id} not found.")
        return []
    cluster = clusters[cluster_id]
    print(f"Processing cluster {cluster_id} with {len(cluster)} stations")
    current_time = datetime.now()
    csv_predictions = {}
    if use_cached_predictions:
        csv_predictions = load_predictions_from_csv()
    for station in cluster:
        name = station['name']
        if use_cached_predictions and name in csv_predictions:
            station['predictions'] = csv_predictions[name]
        else:
            station['predictions'] = get_station_predictions(station, current_time)
        station['current_bikes'] = DEFAULT_STARTING_BIKES
        station['current_payout'] = sum(pred.get('hourly_loss', 0) for pred in station['predictions'])
        # For this example, we assume no adjustment (optimal adjustment = 0)
        station['optimal_adjustment'] = 0
        station['optimal_payout'] = station['current_payout']
        station['payout_benefit'] = station['current_payout'] - station['optimal_payout']
    stations_needing_adjustment = [s for s in cluster if abs(s.get('optimal_adjustment', 0)) > 0]
    sub_clusters = []
    remaining = stations_needing_adjustment.copy()
    while remaining:
        current = remaining.pop(0)
        sub_cluster = [current]
        i = 0
        while i < len(remaining):
            s = remaining[i]
            if calculate_distance((current['latitude'], current['longitude']),
                                  (s['latitude'], s['longitude'])) <= max_distance:
                sub_cluster.append(s)
                remaining.pop(i)
            else:
                i += 1
        total_adjustment = sum(s.get('optimal_adjustment', 0) for s in sub_cluster)
        total_payout_benefit = sum(s.get('payout_benefit', 0) for s in sub_cluster)
        sub_clusters.append({
            'stations': sub_cluster,
            'total_adjustment': total_adjustment,
            'total_payout_benefit': total_payout_benefit,
            'center': calculate_cluster_center(sub_cluster)
        })
    print(f"Created {len(sub_clusters)} sub-clusters")
    return sub_clusters

def optimize_bike_allocation(sub_clusters, available_bikes):
    """
    Optimize bike allocation among sub-clusters.
    (This example uses a simple heuristic based on benefit per bike.)
    """
    for sc in sub_clusters:
        adj = sc['total_adjustment']
        sc['efficiency'] = sc['total_payout_benefit'] / adj if adj > 0 else 0
    sorted_clusters = sorted([sc for sc in sub_clusters if sc['total_adjustment'] > 0],
                               key=lambda x: x['efficiency'], reverse=True)
    allocations = []
    remaining = available_bikes
    total_benefit = 0
    for sc in sorted_clusters:
        needed = sc['total_adjustment']
        bikes_alloc = min(needed, remaining)
        remaining -= bikes_alloc
        benefit = (bikes_alloc / needed) * sc['total_payout_benefit'] if needed else 0
        total_benefit += benefit
        allocations.append({
            'cluster_center': sc['center'],
            'requested_bikes': needed,
            'allocated_bikes': bikes_alloc,
            'efficiency': sc['efficiency'],
            'benefit': benefit
        })
        if remaining <= 0:
            break
    return {'allocations': allocations, 'total_benefit': total_benefit, 'remaining_bikes': remaining}

def export_clusters_to_json(reduced_clusters, allocation=None, filename="reduced_clusters.json"):
    """
    Export reduced clusters (and optional allocation data) to a JSON file.
    """
    if not os.path.exists(EXPORT_DIR):
        os.makedirs(EXPORT_DIR, exist_ok=True)
    output_file = os.path.join(EXPORT_DIR, filename)
    export_data = {
        'timestamp': datetime.now().isoformat(),
        'clusters_count': len(reduced_clusters),
        'clusters': []
    }
    for i, cluster in enumerate(reduced_clusters):
        stations_data = []
        for s in cluster['stations']:
            stations_data.append({
                'name': s['name'],
                'latitude': s['latitude'],
                'longitude': s['longitude'],
                'current_bikes': s.get('current_bikes', 0),
                'optimal_adjustment': s.get('optimal_adjustment', 0),
                'payout_benefit': s.get('payout_benefit', 0)
            })
        cluster_data = {
            'id': i,
            'station_count': len(cluster['stations']),
            'total_adjustment': cluster['total_adjustment'],
            'total_payout_benefit': cluster['total_payout_benefit'],
            'center': cluster['center'],
            'stations': stations_data
        }
        if allocation:
            for alloc in allocation.get('allocations', []):
                if alloc.get('cluster_center') == cluster['center']:
                    cluster_data['allocation'] = alloc
                    break
        export_data['clusters'].append(cluster_data)
    with open(output_file, 'w') as f:
        json.dump(export_data, f, indent=2)
    print(f"Clusters exported to {output_file}")

def load_reduced_clusters_from_json(cluster_id=None, filename="reduced_clusters.json"):
    """
    Load reduced clusters from a JSON file.
    """
    output_file = os.path.join(EXPORT_DIR, filename)
    if not os.path.exists(output_file):
        print(f"Reduced clusters JSON file not found at {output_file}")
        return []
    try:
        with open(output_file, 'r') as f:
            data = json.load(f)
        return data.get('clusters', [])
    except Exception as e:
        print(f"Error loading reduced clusters: {e}")
        return []

def calculate_inter_cluster_travel_costs(reduced_clusters, api_key=GOOGLE_API_KEY):
    """
    Calculate travel costs between reduced clusters using a simple Euclidean distance
    as a proxy. (In a production system you might call a routing API.)
    """
    num_clusters = len(reduced_clusters)
    distance_matrix = {}
    duration_matrix = {}
    cost_matrix = {}
    polyline_matrix = {}
    for i in range(num_clusters):
        cluster_i = reduced_clusters[i]
        cluster_i_name = f"Cluster_{i}"
        distance_matrix[cluster_i_name] = {}
        duration_matrix[cluster_i_name] = {}
        cost_matrix[cluster_i_name] = {}
        polyline_matrix[cluster_i_name] = {}
        origin = cluster_i['center']
        for j in range(num_clusters):
            cluster_j_name = f"Cluster_{j}"
            if i == j:
                distance_matrix[cluster_i_name][cluster_j_name] = 0
                duration_matrix[cluster_i_name][cluster_j_name] = 0
                cost_matrix[cluster_i_name][cluster_j_name] = 0
                polyline_matrix[cluster_i_name][cluster_j_name] = ""
                continue
            dest = reduced_clusters[j]['center']
            dist = calculate_distance(origin, dest)
            duration = dist / 40  # Assume an average speed of 40 km/h
            total_cost = dist * COST_PER_KM + duration * COST_PER_HOUR + dist * WEAR_TEAR_PER_KM
            distance_matrix[cluster_i_name][cluster_j_name] = dist
            duration_matrix[cluster_i_name][cluster_j_name] = duration
            cost_matrix[cluster_i_name][cluster_j_name] = total_cost
            polyline_matrix[cluster_i_name][cluster_j_name] = ""
    return {
        'distance_matrix': distance_matrix,
        'duration_matrix': duration_matrix,
        'cost_matrix': cost_matrix,
        'polyline_matrix': polyline_matrix
    }

def export_travel_costs_to_json(reduced_clusters, filename="cluster_travel_costs.json"):
    """
    Calculate and export travel costs between clusters to a JSON file.
    """
    travel_costs = calculate_inter_cluster_travel_costs(reduced_clusters)
    travel_costs['clusters'] = []
    for i, cluster in enumerate(reduced_clusters):
        cluster_info = {
            'id': i,
            'name': f"Cluster_{i}",
            'center': cluster['center'],
            'station_count': len(cluster['stations']),
            'stations': [s['name'] for s in cluster['stations']],
            'total_adjustment': cluster['total_adjustment'],
            'total_payout_benefit': cluster['total_payout_benefit']
        }
        travel_costs['clusters'].append(cluster_info)
    output_file = os.path.join(EXPORT_DIR, filename)
    with open(output_file, 'w') as f:
        json.dump(travel_costs, f, indent=2)
    print(f"Travel costs exported to {output_file}")
    return travel_costs

def load_travel_costs_from_json(filename="cluster_travel_costs.json"):
    output_file = os.path.join(EXPORT_DIR, filename)
    if not os.path.exists(output_file):
        print(f"Travel costs JSON not found at {output_file}")
        return {}
    try:
        with open(output_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading travel costs JSON: {e}")
        return {}

# ============================
# Visualization Functions
# ============================
def visualize_route(selected_stations, payouts, cost_matrix, title, filename):
    """
    Create a network graph visualization of a route connecting selected stations.
    """
    total_payout = sum(payouts.get(station, 0) for station in selected_stations)
    edges = [(selected_stations[i], selected_stations[i + 1]) for i in range(len(selected_stations) - 1)]
    total_cost = sum(cost_matrix.get(edge[0], {}).get(edge[1], 0) for edge in edges)
    G = nx.Graph()
    for station in selected_stations:
        G.add_node(station, payout=payouts.get(station, 0))
    for edge in edges:
        G.add_edge(edge[0], edge[1], weight=cost_matrix.get(edge[0], {}).get(edge[1], 0))
    pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue', font_size=10, font_weight='bold')
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title(f"{title}\nTotal Payout: {total_payout}, Total Cost: {total_cost}")
    save_path = os.path.join(EXPORT_DIR, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Route visualization saved to {save_path}")

# ============================
# Main Function
# ============================
def main(target_cluster_id=4, use_cached_predictions=True, use_cached_clusters=True,
         calculate_travel_costs_flag=True, use_cached_travel_costs=True):
    # Load clusters with station coordinates
    clusters = load_clusters_with_coordinates()
    if target_cluster_id not in clusters:
        print(f"Cluster {target_cluster_id} not found. Using first available cluster.")
        target_cluster_id = list(clusters.keys())[0]
    current_time = datetime.now()
    # Load predictions (either from CSV or by recalculating)
    if use_cached_predictions:
        csv_preds = load_predictions_from_csv()
        for station in clusters[target_cluster_id]:
            if station['name'] in csv_preds:
                station['predictions'] = csv_preds[station['name']]
            else:
                station['predictions'] = get_station_predictions(station, current_time)
    else:
        for station in clusters[target_cluster_id]:
            station['predictions'] = get_station_predictions(station, current_time)
    export_predictions_to_csv(clusters, target_cluster_id)
    # Load or create reduced clusters
    reduced_clusters = load_reduced_clusters_from_json(target_cluster_id)
    if not reduced_clusters:
        reduced_clusters = create_reduced_clusters(target_cluster_id, clusters=clusters,
                                                    use_cached_predictions=use_cached_predictions)
        export_clusters_to_json(reduced_clusters, allocation=None)
    print("\nReduced Clusters Summary:")
    for i, cluster in enumerate(reduced_clusters):
        print(f"Cluster {i}: {len(cluster['stations'])} stations, Adjustment: {cluster['total_adjustment']}, "
              f"Benefit: ${cluster['total_payout_benefit']:.2f}, Center: {cluster['center']}")
    allocation = None
    if reduced_clusters:
        available_bikes = 20  # Example: 20 bikes available for allocation
        allocation = optimize_bike_allocation(reduced_clusters, available_bikes)
        print(f"\nOptimal Allocation with {available_bikes} bikes: Total Benefit: ${allocation['total_benefit']:.2f}, "
              f"Remaining: {allocation['remaining_bikes']}")
        export_clusters_to_json(reduced_clusters, allocation)
    if reduced_clusters and len(reduced_clusters) > 1 and calculate_travel_costs_flag:
        if use_cached_travel_costs:
            travel_costs = load_travel_costs_from_json()
            if not travel_costs or len(travel_costs.get('clusters', [])) != len(reduced_clusters):
                travel_costs = export_travel_costs_to_json(reduced_clusters)
        else:
            travel_costs = export_travel_costs_to_json(reduced_clusters)
    return "Done"

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Bikeshare Demand and Optimization Pipeline")
    parser.add_argument('--cluster', type=int, default=4, help="Cluster ID to process")
    parser.add_argument('--use-cached-predictions', action='store_true', help="Use cached predictions from CSV")
    parser.add_argument('--use-cached-clusters', action='store_true', help="Use cached reduced clusters from JSON")
    parser.add_argument('--skip-travel-costs', action='store_true', help="Skip calculating travel costs")
    parser.add_argument('--recalculate-travel-costs', action='store_true', help="Force recalculation of travel costs")
    args = parser.parse_args()

    result = main(
        target_cluster_id=args.cluster,
        use_cached_predictions=args.use_cached_predictions,
        use_cached_clusters=args.use_cached_clusters,
        calculate_travel_costs_flag=not args.skip_travel_costs,
        use_cached_travel_costs=not args.recalculate_travel_costs
    )
    print(result)