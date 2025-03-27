import json
import os
from typing import Dict, List, Tuple, Optional
import pandas as pd
from datetime import datetime, timedelta
from enum import Enum
import requests
from dataclasses import dataclass
import time
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.translators import from_docplex_mp
from qiskit_optimization.algorithms import CplexOptimizer, GurobiOptimizer as gp
from docplex.mp.model import Model as DocplexModel
import networkx as nx
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg'
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
plt.ion()  # Enable 
import traceback
import copy
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def load_stations_data() -> Dict[str, Dict[str, any]]:
    """
    Load station data including coordinates, IDs, and weather zones
    based on the existing clusters data
    
    Returns:
    --------
    Dictionary mapping station names to station information
    """
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        coordinates_path = os.path.join(current_dir, "..", "bikeshare-api", "get_clusters", "station_coordinates_cache.json")
        
        # Load coordinates file
        with open(coordinates_path, 'r') as f:
            coordinates_data = json.load(f)
        
        # Convert to standardized station data
        stations = {}
        station_id = 7000  # Starting ID if none is provided
        
        for station_name, coords in coordinates_data.items():
            # Create station entry
            station_info = {
                'station_id': str(station_id),
                'name': station_name,
                'latitude': coords['latitude'],
                'longitude': coords['longitude']
            }
            
            # Use the station name as the key in the dictionary
            stations[station_name] = station_info
            station_id += 1
        
        return stations
        
    except Exception as e:
        print(f"Error loading station data: {e}")
        # Return minimal synthetic data for error recovery
        return {
            'Union Station': {
                'station_id': '7000',
                'name': 'Union Station',
                'latitude': 43.6452,
                'longitude': -79.3806,
            }
        }
def visualize_route(selected_stations, payouts, cost_matrix, title, filename):
    total_payout = sum(payouts[station] for station in selected_stations)
    
    # Create edges based on the selected stations
    edges = []
    for i in range(len(selected_stations) - 1):
        edges.append((selected_stations[i], selected_stations[i + 1]))  # Connect consecutive stations

    total_cost = sum(cost_matrix[edge[0]][edge[1]] for edge in edges if edge[0] in cost_matrix and edge[1] in cost_matrix[edge[0]])
    
    G = nx.Graph()
    for station in selected_stations:
        G.add_node(station, payout=payouts[station])
    
    for edge in edges:
        G.add_edge(edge[0], edge[1], weight=cost_matrix[edge[0]][edge[1]])
    
    pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue', font_size=10, font_weight='bold')
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    
    plt.title(f"{title}\nTotal Payout: {total_payout}, Total Cost: {total_cost}")
    
    # Save the plot to a file
    plt.savefig(filename)  # Save the plot as a file
    plt.close()  # Close the plot to free up memory

# Ensure all enums are defined before use
class Precipitation(str, Enum):
    NONE = "none"
    LIGHT_RAIN = "light rain"
    HEAVY_RAIN = "heavy rain"
    LIGHT_SNOW = "light snow"
    HEAVY_SNOW = "heavy snow"

class Temperature(str, Enum):
    BELOW_MINUS_10 = "below -10°C"
    MINUS_5_TO_0 = "-5°C to 0°C"
    ZERO_TO_5 = "0°C to 5°C"
    FIVE_TO_10 = "5°C to 10°C"
    TEN_TO_15 = "10°C to 15°C"
    FIFTEEN_TO_20 = "15°C to 20°C"
    TWENTY_TO_25 = "20°C to 25°C"
    ABOVE_25 = "above 25°C"

class TTCClosure(str, Enum):
    NONE = "none"
    DELAY_10_20 = "10-20mins"
    DELAY_20_30 = "20-30mins"
    DELAY_30_PLUS = "30+mins"

@dataclass
class WeatherThresholds:
    """Thresholds for categorizing precipitation"""
    LIGHT_RAIN_MM: float = 2.5  # 0-2.5mm per hour is light rain
    HEAVY_RAIN_MM: float = 7.5  # >7.5mm per hour is heavy rain
    LIGHT_SNOW_MM: float = 1.0  # 0-1cm per hour (water equivalent) is light snow
    HEAVY_SNOW_MM: float = 4.0  # >4cm per hour (water equivalent) is heavy snow
    PRECIP_PROBABILITY_THRESHOLD: float = 0.3  # 30% chance threshold

class WeatherService:
    def __init__(self, api_key: str = None):
        """Initialize weather service with API key"""
        self.api_key = '46574182efce52561d5b815bf2c3c5d2'
        if not self.api_key:
            raise ValueError("OpenWeather API key is required")
        
        self.base_url = "http://api.openweathermap.org/data/2.5/forecast"
        self.cache = {}
        self.cache_duration = timedelta(minutes=30)  # Cache weather data for 30 minutes
        self.last_api_call = 0
        self.rate_limit_delay = 1.0  # Seconds between API calls
        
    def _respect_rate_limit(self):
        """Ensure we don't exceed API rate limits"""
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call
        if time_since_last_call < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last_call)
        self.last_api_call = time.time()

    def get_weather_forecast(self, lat: float, lon: float) -> Optional[List[Dict]]:
        """Get 24-hour weather forecast for a location"""
        cache_key = f"{lat:.4f},{lon:.4f}"
        current_time = datetime.now()
        
        # Check cache
        if cache_key in self.cache:
            cached_data, cache_time = self.cache[cache_key]
            if current_time - cache_time < self.cache_duration:
                return cached_data
        
        # Respect rate limiting
        self._respect_rate_limit()
        
        # Make API request
        params = {
            "lat": lat,
            "lon": lon,
            "appid": self.api_key,
            "units": "metric",
            "cnt": 8  # Get 24 hours of data (8 3-hour intervals)
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            if response.status_code == 200:
                data = response.json()
                forecast = self._process_forecast(data)
                self.cache[cache_key] = (forecast, current_time)
                return forecast
            else:
                print(f"Error fetching weather: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error in weather API call: {e}")
            return None

    def _determine_precipitation(self, data: Dict, thresholds: WeatherThresholds = WeatherThresholds()) -> str:
        """
        Determine precipitation type and intensity based on weather data
        """
        # Get precipitation probability
        pop = data.get('pop', 0)  # Probability of precipitation
        
        if pop < thresholds.PRECIP_PROBABILITY_THRESHOLD:
            return "none"
            
        # Get temperature for snow vs rain determination
        temp = data['main']['temp']
        
        # Check rain
        if temp > 0:
            rain_amount = data.get('rain', {}).get('3h', 0) / 3  # Convert 3h to 1h
            if rain_amount > thresholds.HEAVY_RAIN_MM:
                return "heavy rain"
            elif rain_amount > 0:
                return "light rain"
                
        # Check snow
        if temp <= 0:
            snow_amount = data.get('snow', {}).get('3h', 0) / 3  # Convert 3h to 1h
            if snow_amount > thresholds.HEAVY_SNOW_MM:
                return "heavy snow"
            elif snow_amount > 0:
                return "light snow"
                
        return "none"

    def _get_temperature_category(self, temp: float) -> str:
        """Convert temperature to category"""
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

    def _process_forecast(self, data: Dict) -> List[Dict]:
        """Process raw API response into our format"""
        forecasts = []
        
        for item in data['list']:
            temp = item['main']['temp']
            precip = self._determine_precipitation(item)
            
            forecast = {
                'temperature': self._get_temperature_category(temp),
                'precipitation': precip,
                'raw_temp': temp,
                'timestamp': item['dt']
            }
            forecasts.append(forecast)
            
        return forecasts

    def save_cache(self, filename: str = 'weather_cache.json'):
        """Save cache to file"""
        cache_data = {
            key: (data, timestamp.isoformat()) 
            for key, (data, timestamp) in self.cache.items()
        }
        with open(filename, 'w') as f:
            json.dump(cache_data, f)

    def load_cache(self, filename: str = 'weather_cache.json'):
        """Load cache from file"""
        try:
            with open(filename, 'r') as f:
                cache_data = json.load(f)
                self.cache = {
                    key: (data, datetime.fromisoformat(timestamp))
                    for key, (data, timestamp) in cache_data.items()
                }
        except FileNotFoundError:
            self.cache = {}

class DayOfWeek(str, Enum):
    MONDAY = "Monday"
    TUESDAY = "Tuesday"
    WEDNESDAY = "Wednesday"
    THURSDAY = "Thursday"
    FRIDAY = "Friday"
    SATURDAY = "Saturday"
    SUNDAY = "Sunday"

def get_temperature_category(temp: float) -> Temperature:
    """Convert temperature to category"""
    if temp < -10:
        return Temperature.BELOW_MINUS_10
    elif temp < 0:
        return Temperature.MINUS_5_TO_0
    elif temp < 5:
        return Temperature.ZERO_TO_5
    elif temp < 10:
        return Temperature.FIVE_TO_10
    elif temp < 15:
        return Temperature.TEN_TO_15
    elif temp < 20:
        return Temperature.FIFTEEN_TO_20
    elif temp < 25:
        return Temperature.TWENTY_TO_25
    else:
        return Temperature.ABOVE_25

def is_university_in_session(date: datetime) -> bool:
    """
    Determine if universities are in session based on the date.
    
    Academic Calendar (approximate):
    - Fall Term: September to December (excluding holidays)
    - Winter Term: January to April (excluding reading week)
    - Summer Term: May to August (reduced sessions)
    """
    month = date.month
    day = date.day
    
    # Define holiday periods
    winter_break = (month == 12 and day >= 15) or (month == 1 and day <= 8)
    spring_break = month == 2 and 15 <= day <= 21  # Approximate reading week
    summer_break = month in [5, 6, 7, 8]  # Summer has reduced sessions
    winter_exam_period = month == 4 and day >= 15  # Spring exam period
    fall_exam_period = month == 12 and day >= 8  # Fall exam period
    
    # Universities are in full session during fall and winter terms
    if month in [9, 10, 11] or month in [1, 2, 3, 4]:
        # But not during breaks
        if not (winter_break or spring_break or winter_exam_period or fall_exam_period):
            return True
    
    # Summer has reduced sessions
    if summer_break:
        return False
        
    return False

class WeatherZones(Enum):
    """Weather zones for Toronto"""
    TORONTO_CITY_CENTER = "Toronto City Center"
    TORONTO_CITY = "Toronto City"
    TORONTO_INTL_AIRPORT = "Toronto Intl Airport"

@dataclass
class WeatherZoneLocation:
    """Location data for weather zones"""
    zone: WeatherZones
    latitude: float
    longitude: float
    filepath: str
    
    @property
    def coordinates(self):
        return (self.latitude, self.longitude)

# Get the base directory for our project files
def get_base_dir():
    # Start with the current file's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Navigate to the proper directory where data files are stored
    # Move up to REALBIKESTUFF directory
    base_dir = os.path.dirname(os.path.dirname(current_dir))
    
    return base_dir

# Update the WEATHER_ZONE_LOCATIONS with proper path construction
base_dir = get_base_dir()
WEATHER_ZONE_LOCATIONS = [
    WeatherZoneLocation(
        WeatherZones.TORONTO_CITY_CENTER, 
        43.63, 
        -79.4, 
        os.path.join(base_dir,"REALBIKESTUFF", "Region_Creation_Parsing", "Region_Creation_Parsing", "Toronto_City_Center_Weather_03-2024.csv")
    ),
    WeatherZoneLocation(
        WeatherZones.TORONTO_CITY, 
        43.67, 
        -79.40, 
        os.path.join(base_dir,"REALBIKESTUFF", "Region_Creation_Parsing", "Region_Creation_Parsing", "Toronto_Weather-03-2024.csv")
    ),
    WeatherZoneLocation(
        WeatherZones.TORONTO_INTL_AIRPORT, 
        43.68, 
        -79.63, 
        os.path.join(base_dir,"REALBIKESTUFF", "Region_Creation_Parsing", "Region_Creation_Parsing", "TorontoAirportWeather-2024-03.csv")
    )
]

def load_historical_weather_data():
    """Load and process historical weather data from all three zones"""
    weather_data = {}
    
    for zone_location in WEATHER_ZONE_LOCATIONS:
        try:
            # Check if file exists before trying to read it
            if not os.path.exists(zone_location.filepath):
                print(f"Weather data file not found: {zone_location.filepath}")
                print(f"Current working directory: {os.getcwd()}")
                continue
                
            df = pd.read_csv(zone_location.filepath)
            # Clean column names (remove quotes, standardize spacing)
            df.columns = [col.strip().strip('"').strip("'") for col in df.columns]
            
            # Convert date and time to datetime
            df['datetime'] = df['Time (LST)']
            
            # Extract weather conditions for rain and snow
            df['is_rain'] = df['Weather'].fillna('').str.contains('rain', case=False)
            df['is_snow'] = df['Weather'].fillna('').str.contains('snow', case=False)
            df['temperature'] = df['Temp (°C)']
            df['precipitation'] = df['Precip. Amount (mm)']

            
            # Store in dictionary with zone as key
            weather_data[zone_location.zone] = df
            
            print(f"Loaded weather data for {zone_location.zone}: {len(df)} records")
        except Exception as e:
            print(f"Error loading weather data for {zone_location.zone}: {e}")
            print(f"Attempted to load from: {zone_location.filepath}")
    
    return weather_data

def assign_stations_to_zones(stations_data):
    """Assign bike stations to the closest weather zone"""
    # Check if we're getting trip data instead of station data
    if not isinstance(stations_data, dict) or len(stations_data) == 0:
        print("Warning: No valid station data provided to assign_stations_to_zones")
        return {}
        
    # Check the first item to see if it looks like a station
    for station_id, station in stations_data.items():
        if not isinstance(station, dict):
            print(f"Warning: Station {station_id} is not a dictionary")
            continue
            
            
        station_coords = (station['latitude'], station['longitude'])
        closest_zone = None
        min_distance = float('inf')
        
        for zone_location in WEATHER_ZONE_LOCATIONS:
            distance = calculate_distance(station_coords, zone_location.coordinates)
            if distance < min_distance:
                min_distance = distance
                closest_zone = zone_location.zone
        
        station['weather_zone'] = closest_zone
    
    return stations_data

def calculate_distance(coord1, coord2):
    """Calculate Euclidean distance between two coordinate pairs"""
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    return ((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) ** 0.5

def categorize_precipitation(precip_amount, is_rain, is_snow):
    """Categorize precipitation based on amount and type"""
    thresholds = WeatherThresholds()
    
    if is_snow:
        if pd.isna(precip_amount) or precip_amount == 0:
            return Precipitation.NONE
        elif precip_amount <= thresholds.LIGHT_SNOW_MM:
            return Precipitation.LIGHT_SNOW
        else:
            return Precipitation.HEAVY_SNOW
    elif is_rain:
        if pd.isna(precip_amount) or precip_amount == 0:
            return Precipitation.NONE
        elif precip_amount <= thresholds.LIGHT_RAIN_MM:
            return Precipitation.LIGHT_RAIN
        else:
            return Precipitation.HEAVY_RAIN
    else:
        return Precipitation.NONE

def get_temperature_category(temp):
    """Categorize temperature into ranges"""
    if temp < -10:
        return Temperature.BELOW_MINUS_10
    elif temp < 0:
        return Temperature.MINUS_5_TO_0
    elif temp < 5:
        return Temperature.ZERO_TO_5
    elif temp < 10:
        return Temperature.FIVE_TO_10
    elif temp < 15:
        return Temperature.TEN_TO_15
    elif temp < 20:
        return Temperature.FIFTEEN_TO_20
    elif temp < 25:
        return Temperature.TWENTY_TO_25
    else:
        return Temperature.ABOVE_25

def merge_weather_with_historical_data(historical_data, stations_data, weather_data):
    """Merge historical bike data with weather data based on station zone and time"""
    # Ensure we have a dataframe
    if not isinstance(historical_data, pd.DataFrame):
        raise TypeError(f"Expected DataFrame but got {type(historical_data)}")
    
    # Create a proper copy to avoid SettingWithCopyWarning
    result = historical_data.copy()
    
    if 'Start Time' in result.columns:
        result.loc[:, 'Start Time'] = pd.to_datetime(result['Start Time'])
        result.loc[:, 'hour'] = result['Start Time'].dt.hour
        result.loc[:, 'day'] = result['Start Time'].dt.day
        result.loc[:, 'month'] = result['Start Time'].dt.month
        
        # Add Hour column (uppercase) to match what the model expects
        result.loc[:, 'Hour'] = result['hour']
    else:
        print(f"Warning: 'Start Time' column not found in historical data")
        print(f"Available columns: {result.columns.tolist()}")
        # Return data unchanged if we can't process it
        return result
    
    # Add weather columns if they don't exist
    if 'temperature' not in result.columns:
        result.loc[:, 'temperature'] = None
    if 'precipitation' not in result.columns:
        result.loc[:, 'precipitation'] = Precipitation.NONE
    
    # Check if we have valid weather data
    if not weather_data or len(weather_data) == 0:
        print("No weather data available for merging")
        return result
    
    # Sample data for debugging
    print("\n----- SAMPLE DATA FOR DATE DEBUGGING -----")
    if not result.empty:
        sample_row = result.iloc[0]
        print(f"Bike data sample - month: {sample_row['month']}, day: {sample_row['day']}, hour: {sample_row['hour']}")
        print(f"Bike data types - month: {type(sample_row['month'])}, day: {type(sample_row['day'])}, hour: {type(sample_row['hour'])}")
    
    # Process each weather zone data to standardize format
    for zone, zone_data in weather_data.items():
        if zone_data.empty:
            continue
            
        print(f"\nProcessing weather zone: {zone}")
        
        # Print sample of weather data format
        sample_weather = zone_data.iloc[0]
        print(f"Weather data sample - Month: {sample_weather.get('Month')}, Day: {sample_weather.get('Day')}, Time: {sample_weather.get('datetime')}")
        print(f"Weather data types - Month: {type(sample_weather.get('Month'))}, Day: {type(sample_weather.get('Day'))}, Time: {type(sample_weather.get('datetime'))}")
        
        # Ensure Month and Day are numeric for comparison
        try:
            # Convert Month column to numeric if it's not already
            if 'Month' in zone_data.columns and not pd.api.types.is_numeric_dtype(zone_data['Month']):
                zone_data['Month'] = pd.to_numeric(zone_data['Month'].astype(str).str.strip('"').str.strip("'"), errors='coerce')
                print(f"Converted Month column to numeric. Sample: {zone_data['Month'].iloc[0]}")
            
            # Convert Day column to numeric if it's not already
            if 'Day' in zone_data.columns and not pd.api.types.is_numeric_dtype(zone_data['Day']):
                zone_data['Day'] = pd.to_numeric(zone_data['Day'].astype(str).str.strip('"').str.strip("'"), errors='coerce')
                print(f"Converted Day column to numeric. Sample: {zone_data['Day'].iloc[0]}")
                
            # Extract hour from datetime and convert to numeric
            if 'datetime' in zone_data.columns:
                zone_data['extracted_hour'] = zone_data['datetime'].astype(str).str.extract(r'(\d+):', expand=False)
                zone_data['extracted_hour'] = pd.to_numeric(zone_data['extracted_hour'], errors='coerce')
                print(f"Extracted hour column. Sample value: {zone_data['extracted_hour'].iloc[0]}")
        except Exception as e:
            print(f"Error converting weather data columns to numeric: {e}")
            continue
            
        # Update the weather_data dictionary with the processed dataframe
        weather_data[zone] = zone_data
    
    # Counters for diagnostics
    total_rows = len(result)
    matched_rows = 0
    
    # Process each row
    for idx, row in result.iterrows():
        if idx % 100 == 0:
            print(f"Processing row {idx}/{total_rows}")
            
        station_name = row.get('Start Station Name')
        if not station_name or not isinstance(stations_data, dict) or station_name not in stations_data:
            if idx < 10:  # Only print for first few rows to avoid spam
                print(f"Skipping row {idx}: Station '{station_name}' not found in stations_data")
            continue
            
        weather_zone = stations_data[station_name].get('weather_zone')
        if not weather_zone or weather_zone not in weather_data:
            if idx < 10:
                print(f"Skipping row {idx}: Weather zone not found for station '{station_name}'")
            continue
        
        # Get weather for the specific time
        zone_weather = weather_data[weather_zone]
        
        # Find weather record closest to the trip start time
        month, day, hour = row['month'], row['day'], row['hour']
        
        # Display match criteria for debugging (first few rows only)
        if idx < 5:
            print(f"\nTrying to match - month: {month}, day: {day}, hour: {hour}")
        
        # Handle potential data type issues
        try:
            # Use extracted hour column if available
            if 'extracted_hour' in zone_weather.columns:
                time_matches = zone_weather[
                    (zone_weather['Month'] == month) & 
                    (zone_weather['Day'] == day) & 
                    (zone_weather['extracted_hour'] == hour)
                ]
                
                # Report match results for debugging (first few rows only)
                if idx < 5:
                    matching_days = zone_weather[zone_weather['Day'] == day].shape[0]
                    matching_months = zone_weather[zone_weather['Month'] == month].shape[0]
                    matching_hours = zone_weather[zone_weather['extracted_hour'] == hour].shape[0]
                    print(f"Found {matching_days} matching days, {matching_months} matching months, {matching_hours} matching hours")
                    print(f"Full matches found: {time_matches.shape[0]}")
            else:
                # Fallback method
                time_matches = zone_weather[
                    (zone_weather['Month'] == month) & 
                    (zone_weather['Day'] == day)
                ]
                print(f"Warning: No extracted_hour column, matching only on month and day")
        except Exception as e:
            print(f"Error matching weather data for row {idx}: {e}")
            continue
        
        if not time_matches.empty:
            matched_rows += 1
            weather_record = time_matches.iloc[0]
            
            # Update with weather information
            if 'temperature' in weather_record:
                result.at[idx, 'temperature'] = weather_record['temperature']
            elif 'Temp (°C)' in weather_record:
                result.at[idx, 'temperature'] = weather_record['Temp (°C)']
                
            if 'precipitation' in weather_record:
                result.at[idx, 'precipitation'] = weather_record['precipitation']
            elif 'Precip. Amount (mm)' in weather_record:
                result.at[idx, 'precipitation'] = weather_record['Precip. Amount (mm)']
            
            # Determine precipitation type
            is_rain = weather_record.get('is_rain', False)
            is_snow = weather_record.get('is_snow', False)
            precip_amount = result.at[idx, 'precipitation']
            
            # Categorize precipitation and temperature
            result.at[idx, 'precipitation'] = categorize_precipitation(
                precip_amount, is_rain, is_snow
            )
            
            temp_value = result.at[idx, 'temperature']
            if temp_value is not None and not pd.isna(temp_value):
                result.at[idx, 'temperature'] = get_temperature_category(temp_value)
    
    # Print match statistics
    print(f"\nMatched {matched_rows} out of {total_rows} rows ({matched_rows/total_rows*100:.2f}%)")
    
    # Print weather columns to confirm they exist
    print(f"\nAfter weather merge, result contains these columns: {result.columns.tolist()}")
    print(f"Temperature column present: {'temperature' in result.columns}")
    print(f"Precipitation column present: {'precipitation' in result.columns}")
    
    # Show sample of weather data
    if 'temperature' in result.columns and 'precipitation' in result.columns:
        print("\nSample weather data (first 5 rows with non-null values):")
        non_null_temp = result[~result['temperature'].isna()]
        if not non_null_temp.empty:
            for idx, row in non_null_temp.head(5).iterrows():
                print(f"Row {idx}: temperature={row['temperature']}, precipitation={row['precipitation']}")
        else:
            print("No rows with non-null temperature values found")
    
    return result

def prepare_station_data(flows: pd.DataFrame, station_name: str, skip_weather_merge=False) -> pd.DataFrame:
    """
    Prepare features for a specific station from historical flow data
    
    Parameters:
    -----------
    flows : DataFrame with flow data (inflow/outflow marked with flow column)
    station_name : Name of the station
    skip_weather_merge : If True, skip weather merging (assumes weather data already merged)
    
    Returns:
    --------
    DataFrame with prepared features including weather data and user metrics
    """
    # Clean the data before processing
    flows = flows.copy()
    
    # Make sure time columns are datetime
    for col in ['Start Time', 'End Time']:
        if col in flows.columns:
            flows[col] = pd.to_datetime(flows[col])
    
    # Calculate ride duration if Start Time and End Time exist
    if 'Start Time' in flows.columns and 'End Time' in flows.columns:
        flows['ride_duration_minutes'] = (flows['End Time'] - flows['Start Time']).dt.total_seconds() / 60
        # Remove unreasonable durations (negative or >24 hours)
        flows = flows[(flows['ride_duration_minutes'] >= 0) & (flows['ride_duration_minutes'] <= 24*60)]
    
    # Extract user type information
    if 'User Type' in flows.columns:
        # Create separate counts for casual and annual members
        flows['is_casual'] = flows['User Type'].str.contains('Casual', case=False, na=False).astype(int)
        flows['is_annual'] = flows['User Type'].str.contains('Annual', case=False, na=False).astype(int)
    else:
        # Default values if user type isn't available
        flows['is_casual'] = 0
        flows['is_annual'] = 1  # Default to annual member
    
    # Ensure Hour and DayOfWeek exist
    if 'Hour' not in flows.columns:
        # Use appropriate time column based on flow direction
        pos_mask = flows['flow'] > 0  # Inflow
        neg_mask = flows['flow'] < 0  # Outflow
        
        # For inflow, use End Time; for outflow, use Start Time
        flows.loc[pos_mask, 'Hour'] = flows.loc[pos_mask, 'End Time'].dt.hour if 'End Time' in flows.columns else 12
        flows.loc[neg_mask, 'Hour'] = flows.loc[neg_mask, 'Start Time'].dt.hour if 'Start Time' in flows.columns else 12
    
    if 'DayOfWeek' not in flows.columns:
        # Similar logic for DayOfWeek
        pos_mask = flows['flow'] > 0
        neg_mask = flows['flow'] < 0
        
        flows.loc[pos_mask, 'DayOfWeek'] = flows.loc[pos_mask, 'End Time'].dt.day_name() if 'End Time' in flows.columns else 'Monday'
        flows.loc[neg_mask, 'DayOfWeek'] = flows.loc[neg_mask, 'Start Time'].dt.day_name() if 'Start Time' in flows.columns else 'Monday'
    
    # Create directional markers for user types
    flows['casual_outflow'] = ((flows['flow'] < 0) & (flows['is_casual'] == 1)).astype(int)
    flows['casual_inflow'] = ((flows['flow'] > 0) & (flows['is_casual'] == 1)).astype(int)
    flows['annual_outflow'] = ((flows['flow'] < 0) & (flows['is_annual'] == 1)).astype(int)
    flows['annual_inflow'] = ((flows['flow'] > 0) & (flows['is_annual'] == 1)).astype(int)
    
    # Check if we should skip weather merging
    weather_columns_exist = any(col in flows.columns for col in 
                               ['temperature', 'precipitation', 'temp_category', 'precip_category'])
    
    if not skip_weather_merge and not weather_columns_exist:
        # Load weather data
        try:
            # Get coordinates for this station
            stations_data = load_stations_data()
            station_info = next((s for s in stations_data if s['name'] == station_name), None)
            
            if station_info:
                # Determine which weather zone this station belongs to
                station_coords = (station_info['latitude'], station_info['longitude'])
                closest_zone = None
                min_distance = float('inf')
                
                for zone_location in WEATHER_ZONE_LOCATIONS:
                    distance = calculate_distance(station_coords, zone_location.coordinates)
                    if distance < min_distance:
                        min_distance = distance
                        closest_zone = zone_location.zone  # This should be a WeatherZones enum value
                
                if closest_zone is not None:
                    print(f"Assigned station {station_name} to weather zone {closest_zone.name}")
                    
                    # Load all weather data
                    all_weather_data = load_historical_weather_data()
                    
                    # Get weather for this zone
                    if closest_zone in all_weather_data:
                        weather_data = all_weather_data[closest_zone]
                        print(f"Found {len(weather_data)} weather records for zone {closest_zone.name}")
                        
                        # Print weather data columns to debug
                        print(f"Weather data columns: {weather_data.columns.tolist()}")
                        
                        # Create date columns for merging
                        flows['date'] = pd.NA
                        
                        # Use appropriate timestamps based on flow direction
                        pos_mask = flows['flow'] > 0
                        neg_mask = flows['flow'] < 0
                        
                        if 'End Time' in flows.columns:
                            flows.loc[pos_mask, 'date'] = flows.loc[pos_mask, 'End Time'].dt.floor('h')
                        if 'Start Time' in flows.columns:
                            flows.loc[neg_mask, 'date'] = flows.loc[neg_mask, 'Start Time'].dt.floor('h')
                        
                        # Fallback for any missing date values
                        if flows['date'].isna().any():
                            if 'Start Time' in flows.columns:
                                flows.loc[flows['date'].isna(), 'date'] = flows.loc[flows['date'].isna(), 'Start Time'].dt.floor('h')
                            elif 'End Time' in flows.columns:
                                flows.loc[flows['date'].isna(), 'date'] = flows.loc[flows['date'].isna(), 'End Time'].dt.floor('h')
                        
                        # Prepare weather data for merge
                        weather_data['date'] = pd.to_datetime(weather_data['datetime']).dt.floor('h')
                        
                        # Make sure both dataframes have date column as datetime type
                        flows['date'] = pd.to_datetime(flows['date'])
                        weather_data['date'] = pd.to_datetime(weather_data['date'])
                        
                        # Identify the actual temperature and precipitation column names
                        temp_col = next((col for col in weather_data.columns if 'temp' in col.lower()), 'temperature')
                        precip_col = next((col for col in weather_data.columns if 'precip' in col.lower()), 'precipitation')
                        
                        print(f"Using temperature column: {temp_col}")
                        print(f"Using precipitation column: {precip_col}")
                        
                        # Create fallback columns if they don't exist
                        if temp_col not in weather_data.columns:
                            print(f"Warning: Column '{temp_col}' not found in weather data. Creating default.")
                            weather_data[temp_col] = 15.0  # Default temperature
                        
                        if precip_col not in weather_data.columns:
                            print(f"Warning: Column '{precip_col}' not found in weather data. Creating default.")
                            weather_data[precip_col] = 0.0  # Default precipitation
                        
                        # Check for is_rain and is_snow columns
                        if 'is_rain' not in weather_data.columns:
                            weather_data['is_rain'] = weather_data[precip_col] > 0
                        
                        if 'is_snow' not in weather_data.columns:
                            weather_data['is_snow'] = False  # Default to no snow
                        
                        # Merge flows with weather on date
                        try:
                            merged_columns = ['date', temp_col, precip_col, 'is_rain', 'is_snow']
                            available_columns = [col for col in merged_columns if col in weather_data.columns]
                            
                            merged_data = pd.merge(
                                flows, 
                                weather_data[available_columns],
                                on='date', 
                                how='left'
                            )
                            
                            # Rename columns to expected names if they're different
                            if temp_col != 'temperature' and temp_col in merged_data.columns:
                                merged_data['temperature'] = merged_data[temp_col]
                            
                            if precip_col != 'precipitation' and precip_col in merged_data.columns:
                                merged_data['precipitation'] = merged_data[precip_col]
                            
                            print(f"Merge successful: {len(merged_data)} records")
                            return merged_data
                        except Exception as e:
                            print(f"Merge error: {str(e)}")
                            return flows
                    else:
                        print(f"No weather data available for zone {closest_zone.name}")
                else:
                    print(f"Could not determine weather zone for station {station_name}")
            else:
                print(f"Could not find coordinates for station {station_name}")
        except Exception as e:
            print(f"ERROR in prepare_station_data: {str(e)}")
            traceback.print_exc()
        
        # If weather merge failed, return original data
        print("Weather data merge failed, returning original flow data")
    else:
        if skip_weather_merge:
            print(f"Skipping weather merge for station {station_name} as requested")
        else:
            print(f"Weather data already exists for station {station_name}, skipping merge")
    
    return flows

def load_historical_data():
    """Load and combine historical trip data"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_file = os.path.join(base_dir,"Region_Creation_Parsing",  "Region_Creation_Parsing", "Bike share ridership 2024-03.csv")
    
    try:
        # Try different encodings to handle the file correctly
        try:
            # First try with UTF-8 and error handling
            df = pd.read_csv(data_file, encoding='utf-8', errors='replace')
        except UnicodeDecodeError:
            # If that fails, try with cp1252 encoding (common for Windows files)
            df = pd.read_csv(data_file, encoding='cp1252')
        except Exception as e:
            # Final fallback
            df = pd.read_csv(data_file, encoding='latin1')
            
        # Convert timestamps to datetime
        df['Start Time'] = pd.to_datetime(df['Start Time'])
        df['End Time'] = pd.to_datetime(df['End Time'])
        return df
    except FileNotFoundError:
        print(f"Warning: Could not find data file: {data_file}")
        raise
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def load_clusters_with_coordinates():
    """Load clusters and enrich them with station coordinates"""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        clusters_path = os.path.join(current_dir, "..", "Region_Creation_Parsing", "Region_Creation_Parsing", "station_clusters.json")
        coordinates_path = os.path.join(current_dir, "..", "bikeshare-api", "get_clusters", "station_coordinates_cache.json")
        
        # Load files
        with open(clusters_path, 'r') as f:
            clusters_data = json.load(f)  # Changed variable name from _data to clusters_data
        with open(coordinates_path, 'r') as f:
            coordinates_data = json.load(f)
            
        # Clean station names
        cleaned_coords = {name.strip().lower(): coords 
                        for name, coords in coordinates_data.items()}
        
        enriched_clusters = {}
        for cluster_id, station_names in enumerate(clusters_data['clusters']):
            enriched_cluster = []
            for station_name in station_names:
                station_name_clean = station_name.strip().lower()
                if station_name_clean in cleaned_coords:
                    coords = cleaned_coords[station_name_clean]
                    station_info = {
                        'name': station_name,
                        'latitude': coords['latitude'],
                        'longitude': coords['longitude'],
                        'predictions': {}  # Will be filled with weather predictions
                    }
                    enriched_cluster.append(station_info)
                else:
                    print(f"Warning: No coordinates found for station: {station_name}")
            if enriched_cluster:  # Only add clusters with stations
                enriched_clusters[cluster_id] = enriched_cluster
                
        return enriched_clusters
        
    except Exception as e:
        print(f"Error loading clusters and coordinates: {e}")
        raise

def print_cluster_summary(clusters: Dict[int, List[Dict[str, any]]]):
    """Print a summary of the loaded clusters and their stations with predictions"""
    print("\nCluster Summary:")
    print("=" * 50)
    
    for cluster_id, stations in clusters.items():
        print(f"\nCluster {cluster_id}:")
        print(f"Number of stations: {len(stations)}")
        print("Sample stations:")
        for station in stations[:2]:  # Show first 2 stations
            print(f"\n  - {station['name']}")
            print(f"    Location: ({station['latitude']}, {station['longitude']})")
            print("    Next 3 hours predictions:")
            
            # Show first 3 hours of predictions
            for hour, prediction in list(station['predictions'].items())[:3]:
                print(f"      {hour}:")
                print(f"        Demand={prediction['predicted_demand']}, "
                      f"Confidence={prediction['confidence']:.2f}")
                print(f"        Day: {prediction['day_of_week']}, "
                      f"Month: {prediction['month']}")
                print(f"        Weather: {prediction['precipitation']}, "
                      f"Temp: {prediction['temperature']}")
                print(f"        Universities in Session: {prediction['universities_in_session']}")
            
        if len(stations) > 2:
            print(f"\n  ... and {len(stations) - 2} more stations")
        print("-" * 30)

def train_prediction_model(station_name: str) -> Tuple[Dict[str, RandomForestRegressor], Dict[str, LabelEncoder]]:
    """
    Train a prediction model for a specific station with additional metrics
    
    Returns:
    --------
    Tuple containing:
    - Dictionary of models for each metric (flow, ride_duration, user counts)
    - Dictionary of label encoders
    """
    global march_data
    try:
        # Try to use cached preprocessed data with forcenotrefresh=True to bypass validation
        cache_file = preprocess_and_cache_data(forcenotrefresh=True)

        if cache_file and os.path.exists(cache_file):
            # Load pre-merged data from cache
            print("Using preprocessed data cache")
            try:
                march_data = pd.read_csv(cache_file)
                
                # Convert date columns to datetime
                date_columns = ['Start Time', 'End Time', 'date']
                for col in date_columns:
                    if col in march_data.columns:
                        march_data[col] = pd.to_datetime(march_data[col])
            except Exception as e:
                print(f"Failed to read cache file: {e}")
                raise FileNotFoundError("Cache file couldn't be read, loading historical data.")
        else:
            raise FileNotFoundError("Cache file not found, loading historical data.")

    except Exception as e:
        print(f"Error loading cache data: {e}. Falling back to historical data.")

        try:
            if march_data=='None':
                historical_data = load_historical_data()

                # Ensure 'Start Time' column exists and filter for March
                if 'Start Time' in historical_data.columns:
                    historical_data['Start Time'] = pd.to_datetime(historical_data['Start Time'])
                    march_data = historical_data[historical_data['Start Time'].dt.month == 3]
                
                    # If no March data exists, use all historical data
                    if march_data.empty:
                        print("No March data available, using all historical data.")
                        march_data = historical_data
                else:
                    print("No 'Start Time' column found, using all historical data.")
                    march_data = historical_data

                # Load weather and station data
                weather_data = load_historical_weather_data()
                stations_data = load_stations_data()

                # Convert stations_data to dictionary
                stations_dict = {
                    station_name: {
                        'latitude': info.get('latitude'),
                        'longitude': info.get('longitude'),
                        'name': station_name
                    } for station_name, info in stations_data.items()
                }

                print(f"Sample station names: {list(stations_dict.keys())[:5]}")

                # Assign weather zones and merge weather data
                stations_dict = assign_stations_to_zones(stations_dict)
                march_data = merge_weather_with_historical_data(march_data, stations_dict, weather_data)
            
            # Print sample weather values (fixed indentation)
            print("\n===== SAMPLE WEATHER VALUES =====")   
            if 'temperature' in march_data.columns:
                print("First 5 temperature values:")
                print(march_data['temperature'].head(5).tolist())
            elif any('temp' in col.lower() for col in march_data.columns):
                temp_col = next(col for col in march_data.columns if 'temp' in col.lower())
                print(f"First 5 '{temp_col}' values:")
                print(march_data[temp_col].head(5).tolist())            
            else:   
                print("No temperature column found")
        except Exception as e:
            print(f"Error loading or merging data: {e}")
            raise   
        
    # Add day of week before passing to prepare_station_data
    march_data['DayOfWeek'] = march_data['Start Time'].dt.day_name()
    
    try:
        # Calculate flow - split into outflow and inflow parts
        print(f"Looking for station name: '{station_name}'")
        print(f"Sample station names in march_data: {march_data['Start Station Name'].head().tolist()}")

        # Find outflow and inflow trips
        outflow = march_data[march_data['Start Station Name'] == station_name].copy()
        inflow = march_data[march_data['End Station Name'] == station_name].copy()

        print(f"Found {len(outflow)} outflow trips and {len(inflow)} inflow trips for {station_name}")

        # Add hour information and calculate additional metrics
        if len(outflow) > 0:
            outflow['Hour'] = pd.to_datetime(outflow['Start Time']).dt.hour
            outflow['DayOfWeek'] = pd.to_datetime(outflow['Start Time']).dt.day_name()
            outflow['flow'] = -1  # Negative for outflow
            
            # Add user type metrics
            outflow['is_casual'] = (outflow['User Type'] == 'Casual').astype(int)
            outflow['is_annual'] = (outflow['User Type'] == 'Annual').astype(int)
            outflow['casual_outflow'] = outflow['is_casual']
            outflow['casual_inflow'] = 0
            outflow['annual_outflow'] = outflow['is_annual']
            outflow['annual_inflow'] = 0
            
            # Add ride duration if it exists
            if 'Trip  Duration' in outflow.columns:
                outflow['ride_duration_minutes'] = outflow['Trip  Duration'] / 60
        
        if len(inflow) > 0:
            inflow['Hour'] = pd.to_datetime(inflow['End Time']).dt.hour
            inflow['DayOfWeek'] = pd.to_datetime(inflow['End Time']).dt.day_name()
            inflow['flow'] = 1  # Positive for inflow
            
            # Add user type metrics
            inflow['is_casual'] = (inflow['User Type'] == 'Casual').astype(int)
            inflow['is_annual'] = (inflow['User Type'] == 'Annual').astype(int)
            inflow['casual_inflow'] = inflow['is_casual']
            inflow['casual_outflow'] = 0
            inflow['annual_inflow'] = inflow['is_annual']
            inflow['annual_outflow'] = 0
            
            # Add ride duration if it exists
            if 'Trip  Duration' in inflow.columns:
                inflow['ride_duration_minutes'] = inflow['Trip  Duration'] / 60

        # Combine flows
        flows = pd.concat([outflow, inflow]) if len(outflow) > 0 and len(inflow) > 0 else (
            outflow if len(outflow) > 0 else (inflow if len(inflow) > 0 else pd.DataFrame())
        )

        # If we have combined flow data, prepare it
        if not flows.empty:
            # Create outflow and inflow count columns
            flows['outflow_count'] = (flows['flow'] < 0).astype(int)
            flows['inflow_count'] = (flows['flow'] > 0).astype(int)
            
            # Print flow statistics before aggregation
            print("\n===== FLOW STATISTICS BEFORE AGGREGATION =====")
            print(f"Total trips: {len(flows)}")
            print(f"Outflow trips: {flows['outflow_count'].sum()} ({flows['outflow_count'].mean()*100:.1f}%)")
            print(f"Inflow trips: {flows['inflow_count'].sum()} ({flows['inflow_count'].mean()*100:.1f}%)")
            print(f"Net flow: {flows['flow'].sum()}")
            
            # Check if prepare_station_data is modifying our flow values
            original_flow_sum = flows['flow'].sum()
            
            # Add weather data to flows
            station_data = prepare_station_data(flows, station_name, skip_weather_merge=True)
            
            # Group by hour and day to get aggregated values including counts
            if 'flow' in station_data.columns:
                group_cols = ['Hour', 'DayOfWeek']
                
                # Add weather columns if they exist
                if 'temperature' in station_data.columns:
                    group_cols.append('temperature')
                if 'precipitation' in station_data.columns:
                    group_cols.append('precipitation')
                
                # Define aggregation functions - sum for counts, mean for other metrics
                agg_funcs = {
                    'flow': 'sum',
                    'outflow_count': 'sum',
                    'inflow_count': 'sum',
                    'ride_duration_minutes': 'mean',
                    'casual_outflow': 'sum',
                    'casual_inflow': 'sum',
                    'annual_outflow': 'sum',
                    'annual_inflow': 'sum'
                }
                
                # Filter to only columns that exist
                agg_dict = {col: func for col, func in agg_funcs.items() if col in station_data.columns}
                
                # Group with dropna=False to keep all rows
                grouped_data = station_data.groupby(group_cols, dropna=False).agg(agg_dict).reset_index()
                
                # Print flow statistics after aggregation
                print("\n===== FLOW STATISTICS AFTER AGGREGATION =====")
                print(f"Total hours: {len(grouped_data)}")
                if 'outflow_count' in grouped_data.columns:
                    print(f"Hours with outflow: {(grouped_data['outflow_count'] > 0).sum()}")
                    print(f"Max hourly outflow count: {grouped_data['outflow_count'].max()}")
                    print(f"Average hourly outflow: {grouped_data['outflow_count'].mean():.2f}")
                if 'inflow_count' in grouped_data.columns:
                    print(f"Hours with inflow: {(grouped_data['inflow_count'] > 0).sum()}")
                    print(f"Max hourly inflow count: {grouped_data['inflow_count'].max()}")
                    print(f"Average hourly inflow: {grouped_data['inflow_count'].mean():.2f}")
                print(f"Net positive flow hours: {(grouped_data['flow'] > 0).sum()}")
                print(f"Net negative flow hours: {(grouped_data['flow'] < 0).sum()}")
                print(f"Zero flow hours: {(grouped_data['flow'] == 0).sum()}")
                
                if not grouped_data.empty:
                    station_data = grouped_data
    except Exception as e:
        print(f"  DEBUG: Error in data preparation: {e}")
        raise
    
    # Create and fit day-of-week encoder
    try:
        le_day = LabelEncoder()
        le_day.fit(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        
        if 'DayOfWeek' in station_data.columns:
            # Fill any missing values before encoding
            if station_data['DayOfWeek'].isna().any():
                station_data['DayOfWeek'] = station_data['DayOfWeek'].fillna('Monday')
                
            # Make sure all values are valid day names
            valid_days = set(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
            if not set(station_data['DayOfWeek'].unique()).issubset(valid_days):
                print(f"  DEBUG: Found invalid day names: {set(station_data['DayOfWeek'].unique()) - valid_days}")
                # Replace invalid values
                station_data['DayOfWeek'] = station_data['DayOfWeek'].apply(
                    lambda x: x if x in valid_days else 'Monday'
                )
            
            station_data['DayOfWeek_encoded'] = le_day.transform(station_data['DayOfWeek'])
        else:
            print(f"  DEBUG: 'DayOfWeek' column not found in station_data")
            station_data['DayOfWeek_encoded'] = 0  # Default to Monday
    except Exception as e:
        print(f"  DEBUG: Error encoding day of week: {e}")
        raise
        
    # Convert temperature and precipitation to categories
    try:
        def temp_to_category(temp):
            """Convert temperature to category"""
            if temp is None or pd.isna(temp):
                return "unknown"
                
            # Handle numeric temperature
            try:
                temp = float(temp)
                if temp < -10: return "below -10°C"
                elif temp < 0: return "-5°C to 0°C"
                elif temp < 5: return "0°C to 5°C"
                elif temp < 10: return "5°C to 10°C"
                elif temp < 15: return "10°C to 15°C"
                elif temp < 20: return "15°C to 20°C"
                elif temp < 25: return "20°C to 25°C"
                else: return "above 25°C"
            except (ValueError, TypeError):
                # If conversion fails, return unknown
                return "unknown"
        
        def precip_to_category(precip):
            """Convert precipitation to category, handling Enum values and None values"""
            if precip is None or pd.isna(precip):
                return "unknown"
            
            # Handle case when precip is a Precipitation enum
            if isinstance(precip, Precipitation):
                if precip == Precipitation.NONE:
                    return "none"
                elif precip == Precipitation.LIGHT_RAIN:
                    return "light rain"
                elif precip == Precipitation.HEAVY_RAIN:
                    return "heavy rain"
                elif precip == Precipitation.LIGHT_SNOW:
                    return "light snow"
                elif precip == Precipitation.HEAVY_SNOW:
                    return "heavy snow"
                else:
                    return "unknown"
            
            # Handle numeric precipitation values
            try:
                precip_val = float(precip)
                if precip_val == 0:
                    return "none"
                elif precip_val < 2.5:
                    return "light rain"
                else:
                    return "heavy rain"
            except (ValueError, TypeError):
                # If conversion to float fails, return unknown
                return "unknown"
        
        if 'temperature' in station_data.columns:
            station_data['temp_category'] = station_data['temperature'].apply(temp_to_category)
        else:
            print(f"  DEBUG: 'temperature' column not found in station_data")
            station_data['temp_category'] = "15°C to 20°C"  # Default category
        
        if 'precipitation' in station_data.columns:
            station_data['precip_category'] = station_data['precipitation'].apply(precip_to_category)
        else:
            print(f"  DEBUG: 'precipitation' column not found in station_data")
            station_data['precip_category'] = "none"  # Default category
        
        # Initialize encoders with ALL possible values before fitting
        le_temp = LabelEncoder()
        le_temp.fit([
            "below -10°C", "-5°C to 0°C", "0°C to 5°C", "5°C to 10°C", 
            "10°C to 15°C", "15°C to 20°C", "20°C to 25°C", "above 25°C", "unknown"
        ])
        
        le_precip = LabelEncoder()
        le_precip.fit(["none", "light rain", "heavy rain", "light snow", "heavy snow", "unknown"])
        
        # Now transform the data
        station_data['temp_encoded'] = le_temp.transform(station_data['temp_category'])
        station_data['precip_encoded'] = le_precip.transform(station_data['precip_category'])
        
      
    except Exception as e:
        print(f"  DEBUG: Error encoding categorical variables: {e}")
        raise
    
    # Train models for each target variable
    try:
        # Define feature columns
        feature_cols = ['Hour', 'DayOfWeek_encoded', 'temp_encoded', 'precip_encoded']
        
        # Make sure all feature columns exist
        for col in feature_cols:
            if col not in station_data.columns:
                if col == 'Hour':
                    station_data[col] = 12  # Default hour
                else:
                    station_data[col] = 0  # Default encoded value
        
        X = station_data[feature_cols]
        
        # Define target variables to model separately
        target_vars = [
            'flow',                  # Keep this for backward compatibility
            'outflow_count',         # These are new and track actual counts
            'inflow_count',
            'ride_duration_minutes', 
            'casual_outflow', 
            'casual_inflow', 
            'annual_outflow', 
            'annual_inflow'
        ]
        
        # Filter to only existing columns
        available_targets = [var for var in target_vars if var in station_data.columns]
        
        # Create models dictionary
        models = {}
        
        # Train model for each target variable
        for target in available_targets:
            if station_data[target].notna().sum() > 0:  # Check if we have non-NA values
                print(f"\nTraining model for {target}...")
                
                # Basic data statistics
                total_records = len(station_data)
                non_zero = (station_data[target] != 0).sum()
                mean_val = station_data[target].mean()
                
                print(f"Records: {total_records}, Non-zero: {non_zero} ({non_zero/total_records*100:.1f}%)")
                print(f"Min: {station_data[target].min()}, Max: {station_data[target].max()}, Mean: {mean_val:.2f}")
                
                # Split into train/test
                X_train, X_test, y_train, y_test = train_test_split(X, station_data[target], test_size=0.2, random_state=42)
                
                # Choose appropriate model based on target
                if target in ['outflow_count', 'inflow_count']:
                    # Use Gradient Boosting for count prediction (can handle both positive and zero values)
                    from sklearn.ensemble import GradientBoostingRegressor
                    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
                else:
                    # Use Random Forest for other targets
                    from sklearn.ensemble import RandomForestRegressor
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                
                # Train model
                model.fit(X_train, y_train)
                
                # Evaluate on test set
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred) if len(set(y_test)) > 1 else 0
                
                # Print evaluation metrics
                print(f"Test MSE: {mse:.4f}, R²: {r2:.4f}")
                print(f"Predicted - Min: {y_pred.min():.2f}, Max: {y_pred.max():.2f}, Mean: {y_pred.mean():.2f}")
                print(f"Actual - Min: {y_test.min():.2f}, Max: {y_test.max():.2f}, Mean: {y_test.mean():.2f}")
                
                # Store model
                models[target] = model
            else:
                print(f"  DEBUG: No valid data for {target}, using default model")
                # Create a default model
                dummy_X = pd.DataFrame({col: [0] for col in feature_cols})
                dummy_y = [0]
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(n_estimators=10, random_state=42)
                model.fit(dummy_X, dummy_y)
                models[target] = model
        
    except Exception as e:
        print(f"  DEBUG: Error training model: {e}")
        raise
    
    encoders = {
        'day': le_day,
        'temp': le_temp,
        'precip': le_precip
    }
    
    return models, encoders

def get_station_predictions(station: Dict[str, any], hour: datetime) -> List[Dict[str, any]]:
    """Get predictions for a station for the next 24 hours"""
    try:
        # Train models for this station
        models, encoders = train_prediction_model(station['name'])
        
        predictions = []
        current_bikes = 10  # Starting assumption
        
        # Get weather service instance and forecast
        weather_service = WeatherService()
        weather_forecast = weather_service.get_weather_forecast(
            lat=station['latitude'], 
            lon=station['longitude']
        )
        
        if weather_forecast:
            # The weather forecast contains 8 3-hour intervals
            # Expand this to 24 hourly intervals by duplicating each 3-hour entry
            expanded_forecast = []
            for entry in weather_forecast:
                # Each entry covers 3 hours, so duplicate it 3 times
                expanded_forecast.extend([entry] * 3)
            
            # Make sure we have exactly 24 hours (in case the API returned more or less)
            expanded_forecast = expanded_forecast[:24]
            
            # Fill up to 24 hours if needed
            while len(expanded_forecast) < 24:
                # Use the last entry for any missing hours
                if expanded_forecast:
                    expanded_forecast.append(expanded_forecast[-1])
                else:
                    # Default entry if none available
                    expanded_forecast.append({
                        'temperature': "15°C to 20°C",
                        'precipitation': "none"
                    })
            
            weather_forecast = expanded_forecast
        else:
            weather_forecast = []
        
        # Generate predictions for next 24 hours
        for i in range(24):
            prediction_time = hour + timedelta(hours=i)
            prediction_key = prediction_time.strftime('%Y-%m-%d %H:00:00')
            
            # Get appropriate weather data for this hour
            if i < len(weather_forecast):
                forecast_entry = weather_forecast[i]
                temp_category = forecast_entry['temperature']
                precip_category = forecast_entry['precipitation']
            else:
                # Default values if no forecast available
                temp_category = "15°C to 20°C"
                precip_category = "none"
            
            # Prepare features for prediction
            try:
                features = {
                    'Hour': prediction_time.hour,
                    'DayOfWeek_encoded': encoders['day'].transform([prediction_time.strftime('%A')])[0],
                    'temp_encoded': encoders['temp'].transform([temp_category])[0],
                    'precip_encoded': encoders['precip'].transform([precip_category])[0]
                }
            except Exception:
                # Use default features if there's an error
                features = {
                    'Hour': prediction_time.hour,
                    'DayOfWeek_encoded': 0,  # Monday
                    'temp_encoded': 3,       # 15-20°C
                    'precip_encoded': 0      # None
                }
            
            # Convert to DataFrame for prediction
            features_df = pd.DataFrame([features])
            
            # Make prediction for each model
            predictions_dict = {}
            for metric, model in models.items():
                predicted_value = model.predict(features_df)[0]
                predictions_dict[metric] = predicted_value
            
            # Get main flow prediction (or default to 0)
            predicted_flow = predictions_dict.get('flow', 0)
            
            # Update bike count
            new_bikes = max(0, min(25, current_bikes + predicted_flow))
            current_bikes = new_bikes
            
            # Create prediction entry
            pred_entry = {
                'hour': prediction_key,
                'net_flow': round(predicted_flow, 2),
                'predicted_bikes': round(new_bikes),
                'confidence': 0.8,
                'day_of_week': prediction_time.strftime('%A'),
                'temperature': temp_category,
                'precipitation': precip_category,
            }
            
            # Add additional metrics if available
            if 'ride_duration_minutes' in predictions_dict:
                pred_entry['ride_duration'] = round(max(0, predictions_dict['ride_duration_minutes']), 1)
            
            if 'casual_outflow' in predictions_dict:
                pred_entry['casual_outflow'] = round(max(0, predictions_dict['casual_outflow']), 1)
            
            if 'casual_inflow' in predictions_dict:
                pred_entry['casual_inflow'] = round(max(0, predictions_dict['casual_inflow']), 1)
            
            if 'annual_outflow' in predictions_dict:
                pred_entry['annual_outflow'] = round(max(0, predictions_dict['annual_outflow']), 1)
            
            if 'annual_inflow' in predictions_dict:
                pred_entry['annual_inflow'] = round(max(0, predictions_dict['annual_inflow']), 1)
            
            predictions.append(pred_entry)
        
        # Calculate payout metrics using Monte Carlo simulation
        predictions = calculate_station_payout(predictions, station_capacity=25)
        
        return predictions
    except Exception as e:
        traceback.print_exc()
        return []

def preprocess_and_cache_data(force_refresh=False, forcenotrefresh=False):
    """
    Preprocess historical data with weather information and save to cache file.
    
    Args:
        force_refresh: If True, regenerate cache even if it exists
        forcenotrefresh: If True, return cache file path without any validation (overrides force_refresh)
        
    Returns:
        Path to cached data file and station zones dictionary
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(current_dir, "cache")
    cache_file = os.path.join(cache_dir, "merged_historical_weather_data.csv")
    
    # Create cache directory if it doesn't exist
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    # Initialize empty station zones dictionary
    
    # If forcenotrefresh is True, just return the cache file path without validation
    if forcenotrefresh and os.path.exists(cache_file):
        print(f"Using cached data without validation (forcenotrefresh=True)")
        # Try to load station zones but don't fail if it doesn't work
        
        return cache_file
    
    # Check if cache exists and is recent (less than 1 day old)
    cache_exists = os.path.exists(cache_file)
    cache_recent = False
    
    if cache_exists and not force_refresh:
        cache_time = os.path.getmtime(cache_file)
        cache_age = time.time() - cache_time
        cache_recent = cache_age < 86400  # 24 hours in seconds
    
    if cache_exists and cache_recent and not force_refresh:
        print(f"Using cached merged data from {cache_file}")
        
        
        # Test if CSV can be read (just checking, not storing)
        try:
            # Just test if we can read it
            pd.read_csv(cache_file, nrows=5)
            return cache_file
        except Exception as e:
            print(f"Error testing CSV readability: {e}")
            print("Will regenerate cache data")
            force_refresh = True
    
    # Rest of the function remains the same
    # (code for generating/refreshing the cache)

def calculate_station_payout(predictions: List[Dict[str, any]], station_capacity: int = 25) -> List[Dict[str, any]]:
    """
    Calculate payout metrics for each hour based on predicted bike availability
    using optimized Monte Carlo simulation
    """
    import numpy as np
    
    def simulate_pickup_probability(initial_bikes, outflow, inflow, trials=1000):
        """Optimized simulation of pickup probability"""
        if outflow == 0:
            return 0.0
        
        # Pre-generate all random times at once (more efficient)
        arrival_times = np.random.uniform(0, 60, (trials, int(outflow)))
        dropoff_times = np.random.uniform(0, 60, (trials, int(inflow)))
        
        # Sort each trial's times
        arrival_times.sort(axis=1)
        dropoff_times.sort(axis=1)
        
        # Count successful pickups across all trials
        success_count = 0
        
        for trial in range(trials):
            avail_bikes = initial_bikes
            dropoff_idx = 0
            availability = []
            
            for arrival in arrival_times[trial]:
                # Process dropoffs before this arrival
                while dropoff_idx < len(dropoff_times[trial]) and dropoff_times[trial][dropoff_idx] <= arrival:
                    avail_bikes = min(station_capacity, avail_bikes + 1)
                    dropoff_idx += 1
                
                # Record availability
                has_bike = avail_bikes > 0
                availability.append(has_bike)
                
                # Update bike count if pickup occurs
                if has_bike:
                    avail_bikes -= 1
            
            # Select random user experience
            if availability:
                success_count += availability[np.random.randint(0, len(availability))]
        
        return success_count / trials
    
    def simulate_dropoff_probability(initial_bikes, inflow, outflow, trials=1000):
        """Optimized simulation of dropoff probability"""
        if inflow == 0:
            return 0.0
        
        # Pre-generate all random times at once
        dropoff_times = np.random.uniform(0, 60, (trials, int(inflow)))
        pickup_times = np.random.uniform(0, 60, (trials, int(outflow)))
        
        # Sort each trial's times
        dropoff_times.sort(axis=1)
        pickup_times.sort(axis=1)
        
        # Count successful dropoffs across all trials
        success_count = 0
        
        for trial in range(trials):
            bikes_at_station = initial_bikes
            pickup_idx = 0
            space_available = []
            
            for dropoff in dropoff_times[trial]:
                # Process pickups before this dropoff
                while pickup_idx < len(pickup_times[trial]) and pickup_times[trial][pickup_idx] <= dropoff:
                    bikes_at_station = max(0, bikes_at_station - 1)
                    pickup_idx += 1
                
                # Record if space is available
                has_space = bikes_at_station < station_capacity
                space_available.append(has_space)
                
                # Update bike count if dropoff occurs
                if has_space:
                    bikes_at_station += 1
            
            # Select random user experience
            if space_available:
                success_count += space_available[np.random.randint(0, len(space_available))]
        
        return success_count / trials
    
    # Process predictions in batches to calculate payout metrics
    for pred in predictions:
        bikes_available = pred['predicted_bikes']
        
        # Get flow counts with optimized defaults
        casual_outflow = max(0, pred.get('casual_outflow', 0))
        annual_outflow = max(0, pred.get('annual_outflow', 0))
        total_outflow = casual_outflow + annual_outflow
        
        casual_inflow = max(0, pred.get('casual_inflow', 0))
        annual_inflow = max(0, pred.get('annual_inflow', 0))
        total_inflow = casual_inflow + annual_inflow
        
        # If we don't have separate metrics, estimate from net flow
        if total_outflow == 0 and total_inflow == 0:
            net_flow = pred['net_flow']
            if net_flow < 0:
                total_outflow = -net_flow * 1.5
                total_inflow = -net_flow * 0.5
            else:
                total_inflow = net_flow * 1.5
                total_outflow = net_flow * 0.5
        
        # Ensure minimum values for simulation
        total_outflow = max(1, total_outflow)
        total_inflow = max(1, total_inflow)
        
        # Run simulations with fewer trials for speed
        pickup_success_rate = simulate_pickup_probability(
            initial_bikes=bikes_available,
            outflow=total_outflow,
            inflow=total_inflow
        )
        
        dropoff_success_rate = simulate_dropoff_probability(
            initial_bikes=bikes_available,
            inflow=total_inflow,
            outflow=total_outflow
        )
        
        # Calculate metrics
        prob_no_bike = 1 - pickup_success_rate
        prob_no_space = 1 - dropoff_success_rate
        
        missed_casual_pickups = casual_outflow * prob_no_bike
        missed_annual_pickups = annual_outflow * prob_no_bike
        
        missed_casual_dropoffs = casual_inflow * prob_no_space
        missed_annual_dropoffs = annual_inflow * prob_no_space
        BikestoEbikes =0.70; ##70% of trips are regular
        unlock_fee =1;
        E_bike_cost_permin = 0.12;
        Regular_bike_cost_permin = 0.2;
        AveragePerminCost = (E_bike_cost_permin*(1-BikestoEbikes) + BikestoEbikes*Regular_bike_cost_permin);
        AnnualMemberDiscountPenalty = 2/3;
        DropOffFullPenaltyAsFractionOfPickupCost =0.5;

        pickup_loss_casual = missed_casual_pickups*(unlock_fee+ (pred['ride_duration'] * AveragePerminCost));
        ##Casual BIkers make more money per ride than annual members
        pickup_loss_annual = missed_annual_pickups*(unlock_fee+ ( pred['ride_duration'] * AveragePerminCost))*AnnualMemberDiscountPenalty;
        dropoff_loss_casual = missed_casual_dropoffs*(unlock_fee+ (pred['ride_duration'] * AveragePerminCost))*DropOffFullPenaltyAsFractionOfPickupCost;
        dropoff_loss_annual = missed_annual_dropoffs*(unlock_fee+ (pred['ride_duration'] * AveragePerminCost))*DropOffFullPenaltyAsFractionOfPickupCost*AnnualMemberDiscountPenalty;
        total_loss = pickup_loss_casual + pickup_loss_annual + dropoff_loss_casual + dropoff_loss_annual
        
        # Add metrics to prediction
        pred['prob_missed_pickup'] = round(prob_no_bike * 100, 1)
        pred['prob_missed_dropoff'] = round(prob_no_space * 100, 1)
        
        pred['missed_casual_pickups'] = round(missed_casual_pickups, 1)
        pred['missed_annual_pickups'] = round(missed_annual_pickups, 1)
        pred['missed_casual_dropoffs'] = round(missed_casual_dropoffs, 1)
        pred['missed_annual_dropoffs'] = round(missed_annual_dropoffs, 1)
        
        pred['hourly_loss'] = round(total_loss, 2)
    
    return predictions

def print_station_predictions(predictions: List[Dict[str, any]]):
    """Print predictions in a formatted table"""
    if not predictions:
        print("  No predictions available for this station")
        return
    
    # Calculate total metrics
    total_loss = sum(pred.get('hourly_loss', 0) for pred in predictions)
    total_missed_pickups = sum(pred.get('missed_casual_pickups', 0) + 
                              pred.get('missed_annual_pickups', 0) for pred in predictions)
    total_missed_dropoffs = sum(pred.get('missed_casual_dropoffs', 0) + 
                               pred.get('missed_annual_dropoffs', 0) for pred in predictions)
    
    # Print summary
    print(f"\n  24-HOUR STATION SUMMARY:")
    print(f"  Total Predicted Loss: ${total_loss:.2f}")
    print(f"  Total Missed Pickups: {total_missed_pickups:.1f}")
    print(f"  Total Missed Dropoffs: {total_missed_dropoffs:.1f}")
    print("  " + "-" * 80)
    
    # Check which fields are available in the predictions
    sample_pred = predictions[0]
    has_ride_duration = 'ride_duration' in sample_pred
    has_casual = 'casual_outflow' in sample_pred and 'casual_inflow' in sample_pred
    has_annual = 'annual_outflow' in sample_pred and 'annual_inflow' in sample_pred
    has_payout = 'prob_missed_pickup' in sample_pred
    
    # Set up the table width based on available fields
    table_width = 100
    if has_ride_duration:
        table_width += 15
    if has_casual:
        table_width += 30
    if has_annual:
        table_width += 30
    if has_payout:
        table_width += 45
        
    print("\n  Hourly Predictions:")
    print("  " + "-" * table_width)
    
    # Base header
    header = f"  {'Hour':<20} {'Net Flow':>10} {'Predicted Bikes':>15} {'Temperature':>15} {'Precipitation':>15}"
    
    # Add additional columns to header
    if has_ride_duration:
        header += f" {'Ride Duration':>15}"
    if has_casual:
        header += f" {'Casual Out':>12} {'Casual In':>12}"
    if has_annual:
        header += f" {'Annual Out':>12} {'Annual In':>12}"
    if has_payout:
        header += f" {'% Miss Pick':>10} {'% Miss Drop':>10} {'Hourly Loss':>10}"
    
    print(header)
    print("  " + "-" * table_width)
    
    # Print each prediction row
    for pred in predictions:
        # Base row data
        row = f"  {pred['hour']:<20} {pred['net_flow']:>10.2f} {pred['predicted_bikes']:>15.0f} "\
              f"{pred['temperature']:>15} {pred['precipitation']:>15}"
        
        # Add additional columns to row
        if has_ride_duration:
            row += f" {pred.get('ride_duration', 0):>15.1f}"
        if has_casual:
            row += f" {pred.get('casual_outflow', 0):>12.1f} {pred.get('casual_inflow', 0):>12.1f}"
        if has_annual:
            row += f" {pred.get('annual_outflow', 0):>12.1f} {pred.get('annual_inflow', 0):>12.1f}"
        if has_payout:
            row += f" {pred.get('prob_missed_pickup', 0):>10.1f}% {pred.get('prob_missed_dropoff', 0):>10.1f}% ${pred.get('hourly_loss', 0):>9.2f}"
        
        print(row)

def setup_truck_route_optimization():
    # Define stations
    stations = ['Station A', 'Station B', 'Station C', 'Station D']
    
    # Cost matrix representing the distance between each station
    cost_matrix = {
        'Station A': {'Station B': 10, 'Station C': 15, 'Station D': 20},
        'Station B': {'Station A': 10, 'Station C': 25, 'Station D': 30},
        'Station C': {'Station A': 15, 'Station B': 25, 'Station D': 5},
        'Station D': {'Station A': 20, 'Station B': 30, 'Station C': 5}
    }
    
    # Payout for visiting each station
    payouts = {
        'Station A': 100,
        'Station B': 0,  # Not included in optimization
        'Station C': 150,
        'Station D': 200
    }
    
    # Required bikes at each station
    required_bikes = {
        'Station A': 10,
        'Station B': 0,  # Not included in optimization
        'Station C': 20,
        'Station D': 15
    }
    
    # Truck constraints
    truck_capacity = 80
    truck_min_bikes = 0
    starting_bikes = 40
    
    # Initialize the optimization problem
    problem = QuadraticProgram("Truck Route Optimization")
    
    # Add binary variables for each station (1 if visited, 0 if not)
    for station in stations:
        if payouts[station] > 0:  # Only include stations with payout > 0
            problem.binary_var(name=station)

    # Add objective function: maximize profit (payout - cost)
    linear_terms = {}
    quadratic_terms = {}
    
    # Calculate linear terms for payouts
    for station in stations:
        if payouts[station] > 0:
            linear_terms[station] = payouts[station]
    
    # Calculate quadratic terms for costs, excluding Station B
    for station_from in stations:
        if payouts[station_from] > 0:  # Only consider stations with payout
            for station_to in stations:
                if station_from != station_to and payouts[station_to] > 0:
                    cost = cost_matrix[station_from].get(station_to, 0)
                    if cost > 0:
                        quadratic_terms[(station_from, station_to)] = cost
    
    # Set the objective function
    problem.maximize(
        constant=0,
        linear=linear_terms,
        quadratic=quadratic_terms
    )

    # Add constraints for required bikes
    problem.linear_constraint(
        linear={station: required_bikes[station] for station in stations if payouts[station] > 0},
        sense='>=',
        rhs=truck_min_bikes,
        name='min_bikes'
    )
    
    problem.linear_constraint(
        linear={station: required_bikes[station] for station in stations if payouts[station] > 0},
        sense='<=',
        rhs=truck_capacity,
        name='max_bikes'
    )

    # Add constraint for starting bikes
    problem.linear_constraint(
        linear={station: 1 for station in stations if payouts[station] > 0},
        sense='<=',
        rhs=starting_bikes,
        name='starting_bikes'
    )

    return problem

def create_docplex_model():
    # Create a Docplex model
    docplex_model = DocplexModel("Truck Route Optimization")
    
    # Define variables
    vars = {station: docplex_model.binary_var(name=station) for station in ['Station A', 'Station C', 'Station D']}
    
    # Define the objective function
    docplex_model.maximize(
        docplex_model.sum([100 * vars['Station A'], 150 * vars['Station C'], 200 * vars['Station D']]) -
        (125 * vars['Station A'] * vars['Station C'] + 20 * vars['Station A'] * vars['Station D'] +
         5 * vars['Station C'] * vars['Station D'])
    )
    
    # Add constraints
    docplex_model.add_constraint(docplex_model.sum([10 * vars['Station A'], 20 * vars['Station C'], 15 * vars['Station D']]) >= 0, "min_bikes")
    docplex_model.add_constraint(docplex_model.sum([10 * vars['Station A'], 20 * vars['Station C'], 15 * vars['Station D']]) <= 80, "max_bikes")
    
    return docplex_model

def get_hourly_forecast(latitude: float, longitude: float, start_time: datetime) -> Dict[str, Dict[str, any]]:
    """
    Get hourly weather forecast for a specific location starting from the given time
    
    Parameters:
    -----------
    latitude : float
        Latitude of the location
    longitude : float
        Longitude of the location
    start_time : datetime
        Starting time for the forecast
    
    Returns:
    --------
    Dictionary mapping timestamp strings to weather data dictionaries
    """
    
    forecast = {}
    
    try:
        # Get weather service instance
        weather_service = WeatherService()
        
        # Get 24-hour weather forecast
        weather_forecast = weather_service.get_weather_forecast(latitude, longitude)
        
        if weather_forecast:            
            # Process the forecast for the next 24 hours
            for i in range(24):
                forecast_time = start_time + timedelta(hours=i)
                time_key = forecast_time.strftime('%Y-%m-%d %H:00:00')
                
                # Get temperature for this hour
                temperature = weather_service.get_Temperature(latitude, longitude, forecast_time)
                
                # Get precipitation for this hour
                precipitation_type = weather_service.determine_precipitation(latitude, longitude, forecast_time)
                
                # Convert precipitation to string representation
                if precipitation_type == Precipitation.NONE:
                    precip_str = "none"
                elif precipitation_type == Precipitation.LIGHT_RAIN:
                    precip_str = "light rain"
                elif precipitation_type == Precipitation.HEAVY_RAIN:
                    precip_str = "heavy rain"
                elif precipitation_type == Precipitation.LIGHT_SNOW:
                    precip_str = "light snow"
                elif precipitation_type == Precipitation.HEAVY_SNOW:
                    precip_str = "heavy snow"
                else:
                    precip_str = "none"
                
                # Store the forecast for this hour
                forecast[time_key] = {
                    'temperature': temperature,
                    'precipitation': precip_str,
                    'day_of_week': forecast_time.strftime('%A'),
                    'month': forecast_time.strftime('%B')
                }
        else:
            print("DEBUG: Weather forecast returned empty")
    except Exception as e:
        traceback.print_exc()
    
    # If forecast is empty, create a default forecast
    if not forecast:
        for i in range(24):
            forecast_time = start_time + timedelta(hours=i)
            time_key = forecast_time.strftime('%Y-%m-%d %H:00:00')
            
            # Create default values
            forecast[time_key] = {
                'temperature': '15°C to 20°C',
                'precipitation': 'none',
                'day_of_week': forecast_time.strftime('%A'),
                'month': forecast_time.strftime('%B')
            }
    
    return forecast



def calculate_optimal_bike_adjustment(station_predictions: List[Dict[str, any]], 
                                     current_bikes: int = 10,
                                     station_capacity: int = 25) -> Dict[str, any]:
    """
    Calculate the optimal number of bikes to add or remove from a station
    to minimize financial losses over the next 24 hours.
    
    Parameters:
    -----------
    station_predictions: List of hourly predictions for the station
    current_bikes: Current number of bikes at the station
    station_capacity: Maximum capacity of the station
    
    Returns:
    --------
    Dictionary with optimization results
    """
    # Define range of possible adjustments to try
    min_adjustment = -min(current_bikes, 20)  # Don't go below zero, max removal of 20 bikes
    max_adjustment = min(station_capacity - current_bikes, 20)  # Don't exceed capacity, max addition of 20 bikes
    possible_adjustments = range(min_adjustment, max_adjustment + 1)
    
    # Calculate baseline total loss (sum of hourly losses)
    baseline_loss = sum(pred['hourly_loss'] for pred in station_predictions)
    
    best_adjustment = 0
    lowest_loss = baseline_loss
    adjusted_predictions = None
    
    # Try each possible adjustment
    for adjustment in possible_adjustments:
        # Skip testing zero adjustment (already have baseline)
        if adjustment == 0:
            continue
            
        # Create a copy of predictions to simulate with adjusted bike count
        test_predictions = copy.deepcopy(station_predictions)
        
        # Set adjusted initial bike count for first hour
        new_bike_count = current_bikes + adjustment
        
        # Update the bike count for the first hour
        if test_predictions and len(test_predictions) > 0:
            test_predictions[0]['predicted_bikes'] = new_bike_count
        
        # Recalculate flows through all 24 hours
        for i in range(len(test_predictions)):
            # For first hour, use the adjusted count
            if i == 0:
                bikes_available = new_bike_count
            else:
                # For subsequent hours, use previous hour's ending count plus the net flow
                bikes_available = test_predictions[i-1]['predicted_bikes'] + test_predictions[i]['net_flow']
                # Ensure bike count stays within bounds
                bikes_available = max(0, min(station_capacity, bikes_available))
                test_predictions[i]['predicted_bikes'] = bikes_available
        
        # Recalculate the payout metrics with new bike counts
        test_predictions = calculate_station_payout(test_predictions, station_capacity)
        
        # Calculate total loss with this adjustment
        adjusted_loss = sum(pred['hourly_loss'] for pred in test_predictions)
        
        # If this adjustment gives a lower loss, remember it
        if adjusted_loss < lowest_loss:
            lowest_loss = adjusted_loss
            best_adjustment = adjustment
            adjusted_predictions = test_predictions
    
    # Calculate the savings
    savings = baseline_loss - lowest_loss
    
    # Apply the threshold rule: if adjustment is small and savings are low, recommend no change
    if -3 <= best_adjustment <= 3 and savings <= 25:
        best_adjustment = 0
        savings = 0
        
    return {
        'current_bikes': current_bikes,
        'optimal_bikes': current_bikes + best_adjustment,
        'recommended_adjustment': best_adjustment,
        'baseline_loss': round(baseline_loss, 2),
        'optimized_loss': round(lowest_loss, 2),
        'savings': round(savings, 2),
        'optimized_predictions': adjusted_predictions if best_adjustment != 0 else station_predictions
    }

def create_reduced_clusters(cluster_id=3, max_distance=0.5, clusters=None, use_cached_predictions=True):
    """
    Create reduced clusters by combining nearby stations that need bike adjustments.
    
    Args:
        cluster_id: The cluster ID to process (default 3)
        max_distance: Maximum distance in km for stations to be in same sub-cluster (500m = 0.5km)
        clusters: Optional pre-loaded clusters with predictions
        use_cached_predictions: Whether to use already-calculated predictions
        
    Returns:
        List of sub-clusters with their optimal adjustments and payouts
    """
    # Load all clusters if not provided
    if clusters is None:
        all_clusters = load_clusters_with_coordinates()
    else:
        all_clusters = clusters
    
    if cluster_id not in all_clusters:
        print(f"Cluster {cluster_id} not found in cluster data")
        return []
    
    cluster = all_clusters[cluster_id]
    print(f"Processing cluster {cluster_id} with {len(cluster)} stations")
    
    # Get the current time for predictions
    current_time = datetime.now()
    
    # If using cached predictions, try to load from CSV file
    csv_predictions = {}
    if use_cached_predictions:
        csv_predictions = load_predictions_from_csv()
        print(f"Loaded predictions for {len(csv_predictions)} stations from CSV")
    
    # Step 1: Calculate predictions and optimal adjustments for each station
    for station in cluster:
        station_name = station['name']
        
        # Get predictions for the station
        try:
            # Get current predictions (with current bike count)
            current_bikes = 10  # Default - can be replaced with actual value
            station['current_bikes'] = current_bikes
            
            # Check sources for cached predictions in order of preference:
            # 1. First check if station already has predictions (from memory)
            if use_cached_predictions and 'predictions' in station and station['predictions']:
                predictions = station['predictions']
                print(f"Using in-memory predictions for station {station_name}")
            # 2. Next check if we have predictions from CSV
            elif use_cached_predictions and station_name in csv_predictions:
                predictions = csv_predictions[station_name]
                print(f"Using CSV predictions for station {station_name}")
                station['predictions'] = predictions
            # 3. Finally calculate new predictions if necessary
            else:
                print(f"Calculating new predictions for station {station_name}")
                predictions = get_station_predictions(station, current_time)
                station['predictions'] = predictions
            
            # Calculate current payout
            current_payout = sum(pred.get('hourly_loss', 0) for pred in predictions)
            station['current_payout'] = current_payout
            
            # Calculate optimal bike adjustment
            optimization_result = calculate_optimal_bike_adjustment(
                predictions, 
                current_bikes=current_bikes
            )
            
            station['optimal_adjustment'] = optimization_result['recommended_adjustment']
            station['optimal_payout'] = optimization_result['optimized_loss']
            station['adjusted_predictions'] = optimization_result['optimized_predictions']
            
            # Calculate benefit of adjustment
            station['payout_benefit'] = station['current_payout'] - station['optimal_payout']
            
            print(f"Station {station_name}: adjustment={station['optimal_adjustment']}, "
                  f"benefit=${station['payout_benefit']:.2f}")
        except Exception as e:
            print(f"Error processing station {station_name}: {e}")
            traceback.print_exc()
            station['optimal_adjustment'] = 0
            station['payout_benefit'] = 0
    
    # The rest of the function remains the same
    # Step 2: Filter for stations needing adjustment
    stations_needing_adjustment = [s for s in cluster if abs(s.get('optimal_adjustment', 0)) > 0]
    print(f"Found {len(stations_needing_adjustment)} stations needing adjustment")
    
    # Step 3: Create sub-clusters based on proximity
    sub_clusters = []
    remaining_stations = stations_needing_adjustment.copy()
    
    while remaining_stations:
        # Start a new sub-cluster with the first station
        current_station = remaining_stations.pop(0)
        sub_cluster = [current_station]
        
        # Find all stations within max_distance of any station in current sub-cluster
        i = 0
        while i < len(remaining_stations):
            station = remaining_stations[i]
            
            # Check if station is within max_distance of any station in the sub-cluster
            is_nearby = False
            for s in sub_cluster:
                distance = calculate_distance(
                    (s['latitude'], s['longitude']),
                    (station['latitude'], station['longitude'])
                )
                if distance <= max_distance:
                    is_nearby = True
                    break
            
            if is_nearby:
                sub_cluster.append(station)
                remaining_stations.pop(i)
            else:
                i += 1
        
        # Calculate total adjustment and payout for this sub-cluster
        total_adjustment = sum(s['optimal_adjustment'] for s in sub_cluster)
        total_payout_benefit = sum(s['payout_benefit'] for s in sub_cluster)
        
        sub_clusters.append({
            'stations': sub_cluster,
            'total_adjustment': total_adjustment,
            'total_payout_benefit': total_payout_benefit,
            'center': calculate_cluster_center(sub_cluster)
        })
    
    # Print summary of sub-clusters
    print(f"Created {len(sub_clusters)} sub-clusters")
    for i, sc in enumerate(sub_clusters):
        print(f"Sub-cluster {i+1}: {len(sc['stations'])} stations, " 
              f"adjustment: {sc['total_adjustment']}, benefit: ${sc['total_payout_benefit']:.2f}")
        station_names = [s['name'] for s in sc['stations']]
        print(f"  Stations: {', '.join(station_names[:3])}" + 
              (f" and {len(station_names)-3} more" if len(station_names) > 3 else ""))
    
    return sub_clusters

def calculate_distance(point1, point2):
    """
    Calculate the distance between two geographical points
    
    Args:
        point1: Tuple of (latitude, longitude)
        point2: Tuple of (latitude, longitude)
        
    Returns:
        Distance in kilometers
    """
    # Simple Euclidean distance approximation (sufficient for nearby points)
    # 1 degree of latitude ≈ 111 km
    lat_diff = (point1[0] - point2[0]) * 111
    # 1 degree of longitude varies, but at Toronto latitude (43.7°) ≈ 81 km
    lon_diff = (point1[1] - point2[1]) * 81
    
    return (lat_diff**2 + lon_diff**2)**0.5

def calculate_cluster_center(stations):
    """Calculate the geographical center of a cluster of stations"""
    if not stations:
        return (0, 0)
        
    total_lat = sum(s['latitude'] for s in stations)
    total_lon = sum(s['longitude'] for s in stations)
    count = len(stations)
    
    return (total_lat / count, total_lon / count)

def optimize_bike_allocation(sub_clusters, available_bikes):
    """
    Optimize bike allocation across sub-clusters to maximize total payout benefit
    
    Args:
        sub_clusters: List of sub-clusters with their adjustments and payouts
        available_bikes: Number of bikes available for allocation
        
    Returns:
        Dictionary with optimized allocation and expected payout benefit
    """
    # Calculate efficiency (benefit per bike) for each sub-cluster
    for sc in sub_clusters:
        adjustment = sc['total_adjustment']
        if adjustment > 0:  # Only consider clusters that need bikes
            sc['efficiency'] = sc['total_payout_benefit'] / adjustment
        else:
            sc['efficiency'] = 0
    
    # Sort sub-clusters by efficiency (descending)
    sorted_clusters = sorted([sc for sc in sub_clusters if sc['total_adjustment'] > 0], 
                            key=lambda x: x['efficiency'], reverse=True)
    
    # Allocate bikes based on priority
    allocations = []
    remaining_bikes = available_bikes
    total_benefit = 0
    
    for i, cluster in enumerate(sorted_clusters):
        adjustment = cluster['total_adjustment']
        
        # Allocate as many bikes as possible up to what's needed
        bikes_to_allocate = min(adjustment, remaining_bikes)
        remaining_bikes -= bikes_to_allocate
        
        # Calculate proportional benefit based on partial allocation
        benefit = (bikes_to_allocate / adjustment) * cluster['total_payout_benefit']
        total_benefit += benefit
        
        allocations.append({
            'cluster_index': i,
            'stations': [s['name'] for s in cluster['stations']],
            'requested_bikes': adjustment,
            'allocated_bikes': bikes_to_allocate,
            'efficiency': cluster['efficiency'],
            'benefit': benefit
        })
        
        if remaining_bikes <= 0:
            break
    
    return {
        'allocations': allocations,
        'total_benefit': total_benefit,
        'remaining_bikes': remaining_bikes
    }

def load_predictions_from_csv(filename="hourly_predictions.csv"):
    """
    Load station predictions from a previously exported CSV file
    
    Args:
        filename: Name of the CSV file to load
        
    Returns:
        Dictionary mapping station names to their hourly predictions
    """
    import csv
    import os
    
    # Get the path to the CSV file
    export_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exports")
    csv_path = os.path.join(export_dir, filename)
    
    if not os.path.exists(csv_path):
        print(f"Warning: Predictions CSV file not found at {csv_path}")
        return {}
    
    # Dictionary to store predictions by station
    station_predictions = {}
    
    # Read the CSV file
    try:
        with open(csv_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            
            # Process each row
            for row in reader:
                station_name = row['station_name']
                
                # Create a dictionary entry for this station if it doesn't exist
                if station_name not in station_predictions:
                    station_predictions[station_name] = []
                
                # Convert numeric fields from strings
                for field in ['net_flow', 'predicted_bikes', 'casual_outflow', 'casual_inflow', 
                             'annual_outflow', 'annual_inflow', 'ride_duration', 
                             'prob_missed_pickup', 'prob_missed_dropoff', 'hourly_loss']:
                    if field in row:
                        try:
                            row[field] = float(row[field])
                        except (ValueError, TypeError):
                            # If conversion fails, keep as is
                            pass
                
                # Add this hour's prediction to the station's list
                station_predictions[station_name].append(row)
        
        # Sort each station's predictions by hour
        for station, predictions in station_predictions.items():
            station_predictions[station] = sorted(predictions, key=lambda x: x['hour'])
        
        print(f"Successfully loaded predictions for {len(station_predictions)} stations from {csv_path}")
        return station_predictions
        
    except Exception as e:
        print(f"Error loading predictions from CSV: {e}")
        return {}

def export_predictions_to_csv(clusters, cluster_id, filename="hourly_predictions.csv"):
    """
    Export hourly predictions for all stations in a cluster to a CSV file
    
    Args:
        clusters: Dictionary of clusters with station data
        cluster_id: The ID of the cluster to export
        filename: Output filename for the CSV
    """
    import csv
    import os
    
    if cluster_id not in clusters:
        print(f"Cluster {cluster_id} not found for CSV export")
        return
        
    # Create a directory for exports if it doesn't exist
    export_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exports")
    os.makedirs(export_dir, exist_ok=True)
    
    # Full path to the output file
    output_file = os.path.join(export_dir, filename)
    
    # Gather all prediction data
    rows = []
    
    # For each station in the cluster
    for station in clusters[cluster_id]:
        station_name = station['name']
        
        # If this station has predictions
        if 'predictions' in station and station['predictions']:
            # For each hour's prediction
            for pred in station['predictions']:
                # Create a row with all available data
                row = {
                    'station_name': station_name,
                    'station_latitude': station['latitude'],
                    'station_longitude': station['longitude'],
                    'hour': pred.get('hour', ''),
                    'net_flow': pred.get('net_flow', 0),
                    'predicted_bikes': pred.get('predicted_bikes', 0),
                    'temperature': pred.get('temperature', ''),
                    'precipitation': pred.get('precipitation', ''),
                    'day_of_week': pred.get('day_of_week', ''),
                    'casual_outflow': pred.get('casual_outflow', 0),
                    'casual_inflow': pred.get('casual_inflow', 0),
                    'annual_outflow': pred.get('annual_outflow', 0),
                    'annual_inflow': pred.get('annual_inflow', 0),
                    'ride_duration': pred.get('ride_duration', 0),
                    'prob_missed_pickup': pred.get('prob_missed_pickup', 0),
                    'prob_missed_dropoff': pred.get('prob_missed_dropoff', 0),
                    'hourly_loss': pred.get('hourly_loss', 0)
                }
                rows.append(row)
    
    # If we have data, write to CSV
    if rows:
        # Get all field names from the first row
        fieldnames = list(rows[0].keys())
        
        print(f"Writing {len(rows)} prediction records to {output_file}")
        
        # Write the CSV file
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        print(f"CSV export complete. File saved to: {output_file}")
    else:
        print("No prediction data available for CSV export")

def export_clusters_to_json(reduced_clusters, allocation=None, filename="reduced_clusters.json"):
    """
    Export reduced clusters data to a JSON file
    
    Args:
        reduced_clusters: List of reduced cluster data
        allocation: Optional allocation data for these clusters
        filename: Output filename for the JSON
    """
    import json
    import os
    from datetime import datetime
    
    # Create a directory for exports if it doesn't exist
    export_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exports")
    os.makedirs(export_dir, exist_ok=True)
    
    # Full path to the output file
    output_file = os.path.join(export_dir, filename)
    
    # Prepare the export data
    export_data = {
        'timestamp': datetime.now().isoformat(),
        'clusters_count': len(reduced_clusters),
        'clusters': []
    }
    
    # Add each cluster's data
    for i, cluster in enumerate(reduced_clusters):
        # Convert station objects to simplified dicts with just essential info
        stations_data = []
        for station in cluster['stations']:
            station_info = {
                'name': station['name'],
                'latitude': station['latitude'],
                'longitude': station['longitude'],
                'current_bikes': station.get('current_bikes', 0),
                'optimal_adjustment': station.get('optimal_adjustment', 0),
                'payout_benefit': station.get('payout_benefit', 0)
            }
            stations_data.append(station_info)
        
        # Create cluster entry
        cluster_data = {
            'id': i,
            'station_count': len(cluster['stations']),
            'total_adjustment': cluster['total_adjustment'],
            'total_payout_benefit': cluster['total_payout_benefit'],
            'center': cluster['center'],
            'stations': stations_data
        }
        
        # Add allocation data if available
        if allocation and 'allocations' in allocation:
            for alloc in allocation['allocations']:
                if alloc['cluster_index'] == i:
                    cluster_data['allocation'] = {
                        'requested_bikes': alloc['requested_bikes'],
                        'allocated_bikes': alloc['allocated_bikes'],
                        'efficiency': alloc['efficiency'],
                        'benefit': alloc['benefit']
                    }
                    break
        
        export_data['clusters'].append(cluster_data)
    
    # Add overall allocation summary if available
    if allocation:
        export_data['allocation_summary'] = {
            'total_benefit': allocation.get('total_benefit', 0),
            'remaining_bikes': allocation.get('remaining_bikes', 0)
        }
    
    # Write to JSON file
    with open(output_file, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"JSON export complete. File saved to: {output_file}")

def load_reduced_clusters_from_json(cluster_id=None, filename="reduced_clusters.json"):
    """
    Load previously calculated reduced clusters from a JSON file
    
    Args:
        cluster_id: Original cluster ID that was processed (for reference only)
        filename: Name of the JSON file to load
        
    Returns:
        List of reduced clusters
    """
    import json
    import os
    
    # Get the path to the JSON file
    export_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exports")
    json_path = os.path.join(export_dir, filename)
    
    if not os.path.exists(json_path):
        print(f"Warning: Reduced clusters JSON file not found at {json_path}")
        return []
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Extract the reduced clusters from the JSON
        reduced_clusters = []
        
        if 'clusters' in data:
            for cluster_data in data['clusters']:
                # Convert the JSON cluster data back to our internal format
                stations = []
                
                for station_data in cluster_data.get('stations', []):
                    station = {
                        'name': station_data['name'],
                        'latitude': station_data['latitude'],
                        'longitude': station_data['longitude'],
                        'current_bikes': station_data.get('current_bikes', 0),
                        'optimal_adjustment': station_data.get('optimal_adjustment', 0),
                        'payout_benefit': station_data.get('payout_benefit', 0)
                    }
                    stations.append(station)
                
                cluster = {
                    'stations': stations,
                    'center': cluster_data['center'],
                    'total_adjustment': cluster_data['total_adjustment'],
                    'total_payout_benefit': cluster_data['total_payout_benefit']
                }
                
                # Add allocation data if it exists
                if 'allocation' in cluster_data:
                    cluster['allocation'] = cluster_data['allocation']
                
                reduced_clusters.append(cluster)
                
        print(f"Successfully loaded {len(reduced_clusters)} reduced clusters from {json_path}")
        return reduced_clusters
        
    except Exception as e:
        print(f"Error loading reduced clusters from JSON: {e}")
        return []

def calculate_inter_cluster_travel_costs(reduced_clusters, api_key=None):
    """
    Calculate travel costs between reduced clusters using Google Routes API
    
    Args:
        reduced_clusters: List of reduced clusters
        api_key: Google Maps API key (if None, will use default)
        
    Returns:
        Dictionary with travel costs and distance matrix
    """
    import requests
    import json
    from datetime import datetime, timedelta
    import time
    
    # If no API key provided, use the default key
    if api_key is None:
        api_key = "AIzaSyAL7Nxc-3CXVbcwtcO8J6xtdRwWkBr9D48"
    
    # Define cost parameters (these are example values)
    COST_PER_KILOMETER = 0.50  # $0.50 per kilometer for fuel
    COST_PER_HOUR = 45.00      # $25.00 per hour for driver
    WEAR_TEAR_PER_KM = 0.15    # $0.15 per kilometer for vehicle maintenance
    
    # Set up the base URL for the Google Routes API
    base_url = "https://routes.googleapis.com/directions/v2:computeRoutes"
    
    # Setup the headers for the API requests
    headers = {
        'Content-Type': 'application/json',
        'X-Goog-Api-Key': api_key,
        'X-Goog-FieldMask': 'routes.duration,routes.distanceMeters,routes.polyline.encodedPolyline'
    }
    
    # Create a matrix to store distances, durations, and costs
    num_clusters = len(reduced_clusters)
    distance_matrix = {}
    duration_matrix = {}
    cost_matrix = {}
    polyline_matrix = {}
    
    print(f"Calculating travel costs between {num_clusters} clusters...")
    
    # For each pair of clusters, calculate the route
    for i in range(num_clusters):
        cluster_i = reduced_clusters[i]
        cluster_i_name = f"Cluster_{i}"
        
        # Initialize matrices for this cluster
        distance_matrix[cluster_i_name] = {}
        duration_matrix[cluster_i_name] = {}
        cost_matrix[cluster_i_name] = {}
        polyline_matrix[cluster_i_name] = {}
        
        # Get the center coordinates of cluster i
        origin_lat = cluster_i['center'][0]
        origin_lng = cluster_i['center'][1]
        
        for j in range(num_clusters):
            # Skip if it's the same cluster
            if i == j:
                distance_matrix[cluster_i_name][f"Cluster_{j}"] = 0
                duration_matrix[cluster_i_name][f"Cluster_{j}"] = 0
                cost_matrix[cluster_i_name][f"Cluster_{j}"] = 0
                polyline_matrix[cluster_i_name][f"Cluster_{j}"] = ""
                continue
                
            cluster_j = reduced_clusters[j]
            
            # Get the center coordinates of cluster j
            dest_lat = cluster_j['center'][0]
            dest_lng = cluster_j['center'][1]
            
            # Create the request body
            request_body = {
                "origin": {
                    "location": {
                        "latLng": {
                            "latitude": origin_lat,
                            "longitude": origin_lng
                        }
                    }
                },
                "destination": {
                    "location": {
                        "latLng": {
                            "latitude": dest_lat,
                            "longitude": dest_lng
                        }
                    }
                },
                "travelMode": "DRIVE",
                "routingPreference": "TRAFFIC_AWARE",
                "computeAlternativeRoutes": False,
                "routeModifiers": {
                    "avoidTolls": False,
                    "avoidHighways": False,
                    "avoidFerries": False
                },
                "languageCode": "en-US",
                "units": "METRIC"
            }
            
            # Make the API request
            try:
                print(f"Requesting route from Cluster {i} to Cluster {j}...")
                response = requests.post(
                    base_url, 
                    headers=headers, 
                    json=request_body
                )
                
                # Check for successful response
                if response.status_code == 200:
                    route_data = response.json()
                    
                    # Extract distance and duration from the response
                    if 'routes' in route_data and len(route_data['routes']) > 0:
                        route = route_data['routes'][0]
                        
                        # Get distance in kilometers
                        distance_meters = route.get('distanceMeters', 0)
                        distance_km = distance_meters / 1000.0
                        
                        # Get duration in hours (comes as a string like "165s" or "10m5s")
                        duration_str = route.get('duration', '0s')
                        
                        # Parse the duration string to get seconds
                        duration_seconds = 0
                        if 'h' in duration_str:
                            hours, rest = duration_str.split('h', 1)
                            duration_seconds += int(hours) * 3600
                            duration_str = rest
                        if 'm' in duration_str:
                            minutes, rest = duration_str.split('m', 1)
                            duration_seconds += int(minutes) * 60
                            duration_str = rest
                        if 's' in duration_str:
                            seconds = duration_str.rstrip('s')
                            if seconds:  # Make sure it's not an empty string
                                duration_seconds += int(seconds)
                        
                        duration_hours = duration_seconds / 3600.0
                        
                        # Calculate costs
                        fuel_cost = distance_km * COST_PER_KILOMETER
                        driver_cost = duration_hours * COST_PER_HOUR
                        maintenance_cost = distance_km * WEAR_TEAR_PER_KM
                        total_cost = fuel_cost + driver_cost + maintenance_cost
                        
                        # Get the polyline
                        polyline = route.get('polyline', {}).get('encodedPolyline', "")
                        
                        # Store in matrices
                        distance_matrix[cluster_i_name][f"Cluster_{j}"] = distance_km
                        duration_matrix[cluster_i_name][f"Cluster_{j}"] = duration_hours
                        cost_matrix[cluster_i_name][f"Cluster_{j}"] = total_cost
                        polyline_matrix[cluster_i_name][f"Cluster_{j}"] = polyline
                        
                        print(f"  Distance: {distance_km:.2f} km, Duration: {duration_hours:.2f} hours, Cost: ${total_cost:.2f}")
                    else:
                        print(f"  No routes found between Cluster {i} and Cluster {j}")
                        # Set default values
                        distance_matrix[cluster_i_name][f"Cluster_{j}"] = 0
                        duration_matrix[cluster_i_name][f"Cluster_{j}"] = 0
                        cost_matrix[cluster_i_name][f"Cluster_{j}"] = 0
                        polyline_matrix[cluster_i_name][f"Cluster_{j}"] = ""
                else:
                    print(f"  API request failed with status code {response.status_code}: {response.text}")
                    # Set default values
                    distance_matrix[cluster_i_name][f"Cluster_{j}"] = 0
                    duration_matrix[cluster_i_name][f"Cluster_{j}"] = 0
                    cost_matrix[cluster_i_name][f"Cluster_{j}"] = 0
                    polyline_matrix[cluster_i_name][f"Cluster_{j}"] = ""
                
                # Add a small delay to avoid hitting API rate limits
                time.sleep(0.5)
                
            except Exception as e:
                print(f"  Error calculating route between Cluster {i} and Cluster {j}: {e}")
                # Set default values
                distance_matrix[cluster_i_name][f"Cluster_{j}"] = 0
                duration_matrix[cluster_i_name][f"Cluster_{j}"] = 0
                cost_matrix[cluster_i_name][f"Cluster_{j}"] = 0
                polyline_matrix[cluster_i_name][f"Cluster_{j}"] = ""
    
    # Compile results
    result = {
        'timestamp': datetime.now().isoformat(),
        'cost_parameters': {
            'fuel_cost_per_km': COST_PER_KILOMETER,
            'driver_cost_per_hour': COST_PER_HOUR,
            'maintenance_cost_per_km': WEAR_TEAR_PER_KM
        },
        'distance_matrix': distance_matrix,
        'duration_matrix': duration_matrix,
        'cost_matrix': cost_matrix,
        'polyline_matrix': polyline_matrix
    }
    
    return result

def export_travel_costs_to_json(reduced_clusters, filename="cluster_travel_costs.json"):
    """
    Calculate and export travel costs between clusters to a JSON file
    
    Args:
        reduced_clusters: List of reduced clusters
        filename: Output filename for the JSON
    """
    import json
    import os
    
    # Create a directory for exports if it doesn't exist
    export_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exports")
    os.makedirs(export_dir, exist_ok=True)
    
    # Full path to the output file
    output_file = os.path.join(export_dir, filename)
    
    # Calculate travel costs
    print("\n=== Calculating travel costs between clusters ===")
    travel_costs = calculate_inter_cluster_travel_costs(reduced_clusters)
    
    # Add cluster information to the results
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
    
    # Write to JSON file
    with open(output_file, 'w') as f:
        json.dump(travel_costs, f, indent=2)
    
    print(f"Travel costs export complete. File saved to: {output_file}")
    return travel_costs

def load_travel_costs_from_json(filename="cluster_travel_costs.json"):
    """
    Load previously calculated travel costs between clusters from a JSON file
    
    Args:
        filename: Name of the JSON file to load
        
    Returns:
        Dictionary with travel costs data
    """
    import json
    import os
    
    # Get the path to the JSON file
    export_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exports")
    json_path = os.path.join(export_dir, filename)
    
    if not os.path.exists(json_path):
        print(f"Warning: Travel costs JSON file not found at {json_path}")
        return {}
    
    try:
        with open(json_path, 'r') as f:
            travel_costs = json.load(f)
        
        print(f"Successfully loaded travel costs for {len(travel_costs.get('clusters', []))} clusters from {json_path}")
        return travel_costs
        
    except Exception as e:
        print(f"Error loading travel costs from JSON: {e}")
        return {}

def main(target_cluster_id=3, use_cached_predictions=True, use_cached_clusters=True, 
         calculate_travel_costs=True, use_cached_travel_costs=True):
    """
    Main function to run the station prediction and cluster optimization pipeline
    
    Args:
        target_cluster_id: ID of the cluster to process
        use_cached_predictions: Whether to use cached predictions from CSV instead of recalculating
        use_cached_clusters: Whether to use cached reduced clusters from JSON instead of recalculating
        calculate_travel_costs: Whether to calculate travel costs between clusters
        use_cached_travel_costs: Whether to use cached travel costs from JSON
        
    Returns:
        Status message
    """
    # Load clusters with coordinates
    clusters = load_clusters_with_coordinates()
    
    # Check if the target cluster exists
    if target_cluster_id not in clusters:
        print(f"Error: Cluster {target_cluster_id} not found. Available clusters: {list(clusters.keys())}")
        # If target cluster doesn't exist, get the first available cluster
        target_cluster_id = list(clusters.keys())[0]
        print(f"Using cluster {target_cluster_id} instead.")
    
    # Get the current time for predictions
    current_time = datetime.now()
    
    # Load predictions from CSV if requested
    if use_cached_predictions:
        print(f"\n=== Loading cached predictions from CSV ===")
        csv_predictions = load_predictions_from_csv()
        
        if csv_predictions:
            print(f"Loaded predictions for {len(csv_predictions)} stations")
            
            # Assign predictions to stations in the cluster
            for station in clusters[target_cluster_id]:
                if station['name'] in csv_predictions:
                    station['predictions'] = csv_predictions[station['name']]
                    print(f"Assigned cached predictions to station {station['name']}")
                else:
                    print(f"Warning: No cached predictions found for station {station['name']}")
    
    # If we're not using cached predictions or some stations don't have predictions, calculate them
    if not use_cached_predictions:
        print(f"\n=== Processing Cluster {target_cluster_id} ===")
        print(f"Number of stations: {len(clusters[target_cluster_id])}")
        print(f"Generating predictions for next 24 hours starting from {current_time.strftime('%Y-%m-%d %H:%M')}")
        print("=" * 80)
        
        # Process all stations in the target cluster
        for station in clusters[target_cluster_id]:
            # Skip if we already have predictions for this station
            if use_cached_predictions and 'predictions' in station:
                print(f"\nStation: {station['name']} - Using cached predictions")
                continue
                
            print(f"\nStation: {station['name']}")
            print(f"Location: ({station['latitude']}, {station['longitude']})")
            
            # Get predictions for the next 24 hours
            predictions = get_station_predictions(station, current_time)
            
            # Calculate station payout
            predictions = calculate_station_payout(predictions)
            
            # Store the predictions in the station object for CSV export and reuse
            station['predictions'] = predictions
            
            # Print the predictions
            print_station_predictions(predictions)
            print("-" * 80)
    
        # Export all predictions to CSV if we calculated new ones
        print("\n=== Exporting Hourly Predictions to CSV ===")
        export_predictions_to_csv(clusters, target_cluster_id)
    
    # Load reduced clusters from JSON if requested
    reduced_clusters = []
    if use_cached_clusters:
        print(f"\n=== Loading cached reduced clusters from JSON ===")
        reduced_clusters = load_reduced_clusters_from_json(target_cluster_id)
        
        if not reduced_clusters:
            print("No cached reduced clusters found, will calculate them instead")
            use_cached_clusters = False
    
    # Calculate reduced clusters if needed
    if not use_cached_clusters:
        # Test the create_reduced_clusters function
        print("\n\n=== Creating reduced clusters ===")
        print("=" * 80)
        # Pass the clusters with cached predictions to avoid recalculating
        reduced_clusters = create_reduced_clusters(target_cluster_id, clusters=clusters, use_cached_predictions=True)
        
        # Export reduced clusters to JSON
        print("\n=== Exporting Reduced Clusters to JSON ===")
        export_clusters_to_json(reduced_clusters, None)  # None for allocation, will be calculated below
    
    # Print summary of reduced clusters
    print("\nReduced Clusters Summary:")
    for i, cluster in enumerate(reduced_clusters):
        print(f"Cluster {i}: {len(cluster['stations'])} stations")
        print(f"  Total adjustment: {cluster['total_adjustment']} bikes")
        print(f"  Total benefit: ${cluster['total_payout_benefit']:.2f}")
        print(f"  Center location: {cluster['center']}")
        print("-" * 50)
    
    allocation = None
    
    # If we have reduced clusters, test the bike allocation optimization
    if reduced_clusters:
        print("\n=== Testing bike allocation optimization ===")
        available_bikes = 20  # Example: 20 bikes available for allocation
        allocation = optimize_bike_allocation(reduced_clusters, available_bikes)
        
        print(f"\nOptimal Allocation with {available_bikes} available bikes:")
        print(f"Total expected benefit: ${allocation['total_benefit']:.2f}")
        print(f"Remaining bikes: {allocation['remaining_bikes']}")
        
        print("\nAllocation by cluster:")
        for alloc in allocation['allocations']:
            print(f"Cluster {alloc['cluster_index']}: {alloc['allocated_bikes']}/{alloc['requested_bikes']} bikes")
            print(f"  Expected benefit: ${alloc['benefit']:.2f}")
            print(f"  Efficiency: ${alloc['efficiency']:.2f} per bike")
            print(f"  Stations: {', '.join(alloc['stations'][:3])}" + 
                 (f" and {len(alloc['stations'])-3} more" if len(alloc['stations']) > 3 else ""))
            print("-" * 40)
    
        # Re-export reduced clusters with allocation data
        print("\n=== Exporting Reduced Clusters with Allocation to JSON ===")
        export_clusters_to_json(reduced_clusters, allocation)
    
    # Export travel costs data to JSON if we have reduced clusters and it's requested
    if reduced_clusters and len(reduced_clusters) > 1 and calculate_travel_costs:
        # Check if we should use cached travel costs
        if use_cached_travel_costs:
            print("\n=== Loading cached travel costs between clusters ===")
            travel_costs = load_travel_costs_from_json()
            
            if travel_costs and 'clusters' in travel_costs and len(travel_costs['clusters']) == len(reduced_clusters):
                print(f"Using cached travel costs for {len(travel_costs['clusters'])} clusters")
            else:
                print("No matching cached travel costs found, will calculate them instead")
                print("\n=== Calculating and exporting travel costs between clusters ===")
                export_travel_costs_to_json(reduced_clusters)
        else:
            print("\n=== Calculating and exporting travel costs between clusters ===")
            export_travel_costs_to_json(reduced_clusters)
    
    return "Done"

if __name__ == "__main__":
    import argparse
    
    # Create argument parser
    parser = argparse.ArgumentParser(description='Predict bikeshare demand and optimize bike distribution')
    parser.add_argument('--cluster', type=int, default=3, help='Cluster ID to process (default: 3)')
    parser.add_argument('--use-cached-predictions', action='store_true', help='Use cached predictions from CSV')
    parser.add_argument('--use-cached-clusters', action='store_true', help='Use cached reduced clusters from JSON')
    parser.add_argument('--skip-travel-costs', action='store_true', help='Skip calculating travel costs')
    parser.add_argument('--recalculate-travel-costs', action='store_true', help='Force recalculation of travel costs')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run main function with arguments
    result = main(
        target_cluster_id=args.cluster,
        use_cached_predictions=args.use_cached_predictions,
        use_cached_clusters=args.use_cached_clusters,
        calculate_travel_costs=not args.skip_travel_costs,
        use_cached_travel_costs=not args.recalculate_travel_costs
    )
    
    print(result)
    
    # Note: The code below is for testing other optimization approaches
     # Setup Qiskit optimization problem
    print(main())
    start_time = time.time()
    qiskit_problem = setup_truck_route_optimization()
    qiskit_runtime = time.time() - start_time
    print("Qiskit Optimization Problem:")
    print(qiskit_problem.prettyprint())
    print(f"Qiskit Runtime: {qiskit_runtime:.4f} seconds")

    # Setup Docplex model
    start_time = time.time()
    docplex_model = create_docplex_model()
    docplex_runtime = time.time() - start_time
    print("\nDocplex Model:")
    print(docplex_model.prettyprint())
    print(f"Docplex Runtime: {docplex_runtime:.4f} seconds")
    cost_matrix = {
        'Station A': {'Station B': 10, 'Station C': 15, 'Station D': 20},
        'Station B': {'Station A': 10, 'Station C': 25, 'Station D': 30},
        'Station C': {'Station A': 15, 'Station B': 25, 'Station D': 5},
        'Station D': {'Station A': 20, 'Station B': 30, 'Station C': 5}
    }
    payouts = {'Station A': 100, 'Station B': 0, 'Station C': 150, 'Station D': 200}
    
    # Quantum Program Optimizer results
    qiskit_selected_stations = ['Station A', 'Station C', 'Station D']
    
    # Visualize and save both results
    print("Generating Quantum Optimizer Visualization...")
    visualize_route(qiskit_selected_stations, payouts, cost_matrix, "Quantum Program Optimizer Route", "quantum_optimizer_route.png")
    
    # Docplex Optimizer results
    docplex_selected_stations = ['Station A', 'Station D', 'Station C']
    
    print("Generating Docplex Optimizer Visualization...")
    visualize_route(docplex_selected_stations, payouts, cost_matrix, "Docplex Optimizer Route", "docplex_optimizer_route.png")