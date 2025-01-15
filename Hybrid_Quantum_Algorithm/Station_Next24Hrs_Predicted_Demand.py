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
    LIGHT_SNOW_MM: float = 1.0  # 0-1mm per hour (water equivalent) is light snow
    HEAVY_SNOW_MM: float = 4.0  # >4mm per hour (water equivalent) is heavy snow
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

class WeatherZones:
    """Manages weather zones for Toronto bike stations"""
    
    # Define 6 strategic locations covering Toronto's bike share network
    WEATHER_ZONES = {
        'downtown_core': {'lat': 43.6547, 'lon': -79.3815},  # Downtown Toronto
        'north_york': {'lat': 43.7615, 'lon': -79.4111},     # North York
        'east_end': {'lat': 43.6772, 'lon': -79.3300},       # East Toronto
        'west_end': {'lat': 43.6427, 'lon': -79.4285},       # West Toronto
        'midtown': {'lat': 43.6889, 'lon': -79.3833},        # Midtown
        'liberty_village': {'lat': 43.6371, 'lon': -79.4187}  # Liberty Village
    }

    def __init__(self):
        self.zone_assignments = {}
        self.weather_data = {}

    def assign_stations_to_zones(self, stations_coords: Dict[str, Dict[str, float]]) -> Dict[str, str]:
        """
        Assign each station to the nearest weather zone
        
        Args:
            stations_coords: Dictionary of station names and their coordinates
            
        Returns:
            Dictionary mapping station names to their assigned weather zone
        """
        zone_locations = np.array([[z['lat'], z['lon']] for z in self.WEATHER_ZONES.values()])
        zone_names = list(self.WEATHER_ZONES.keys())
        
        assignments = {}
        for station_name, coords in stations_coords.items():
            station_location = np.array([coords['latitude'], coords['longitude']])
            
            # Calculate distances to all zones
            distances = np.sqrt(np.sum((zone_locations - station_location) ** 2, axis=1))
            
            # Assign to nearest zone
            nearest_zone_idx = np.argmin(distances)
            assignments[station_name] = zone_names[nearest_zone_idx]
        
        self.zone_assignments = assignments
        return assignments

    def get_station_weather_zone(self, station_name: str) -> Dict[str, float]:
        """Get the weather zone coordinates for a given station"""
        if station_name not in self.zone_assignments:
            raise ValueError(f"Station {station_name} has not been assigned to a weather zone")
            
        zone_name = self.zone_assignments[station_name]
        return self.WEATHER_ZONES[zone_name]

def load_clusters_with_coordinates() -> Dict[int, List[Dict[str, any]]]:
    """
    Load clusters and enrich them with station coordinates and 24-hour predictions.
    Uses weather zones to minimize API calls.
    """
    try:
        # Initialize services
        weather_service = WeatherService()
        weather_service.load_cache()
        weather_zones = WeatherZones()
        
        # Load data files
        current_dir = os.path.dirname(os.path.abspath(__file__))
        clusters_path = os.path.join(current_dir, "..", "Region_Creation_Parsing", "station_clusters.json")
        coordinates_path = os.path.join(current_dir, "..", "bikeshare-api", "get_clusters", "station_coordinates_cache.json")
        
        with open(clusters_path, 'r') as f:
            clusters_data = json.load(f)
        with open(coordinates_path, 'r') as f:
            coordinates_data = json.load(f)
            
        # Assign stations to weather zones
        weather_zones.assign_stations_to_zones(coordinates_data)
        
        # Get weather forecasts for each zone (only 6 API calls)
        zone_forecasts = {}
        for zone_name, coords in WeatherZones.WEATHER_ZONES.items():
            forecast = weather_service.get_weather_forecast(coords['lat'], coords['lon'])
            if forecast:
                zone_forecasts[zone_name] = forecast
            else:
                print(f"Warning: Could not get weather forecast for zone {zone_name}")
        
        # Get current time and generate next 24 hours
        current_time = datetime.now()
        next_24_hours = [
            current_time + timedelta(hours=i)
            for i in range(24)
        ]
        
        enriched_clusters: Dict[int, List[Dict[str, any]]] = {}
        
        for cluster_id, station_names in enumerate(clusters_data['clusters']):
            enriched_cluster = []
            for station_name in station_names:
                if station_name in coordinates_data:
                    try:
                        # Get station's weather zone
                        zone_name = weather_zones.zone_assignments[station_name]
                        weather_forecast = zone_forecasts.get(zone_name)
                        
                        station_info = {
                            'name': station_name,
                            'latitude': coordinates_data[station_name]['latitude'],
                            'longitude': coordinates_data[station_name]['longitude'],
                            'weather_zone': zone_name,
                            'current_time': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                            'predictions': {}
                        }
                        
                        # Add 24 hourly predictions using zone's weather
                        for hour in next_24_hours:
                            hour_str = hour.strftime('%Y-%m-%d %H:00:00')
                            
                            # Find closest weather forecast for the zone
                            weather = None
                            if weather_forecast:
                                hour_timestamp = hour.timestamp()
                                weather = min(weather_forecast, 
                                           key=lambda x: abs(x['timestamp'] - hour_timestamp))
                            
                            station_info['predictions'][hour_str] = {
                                'predicted_demand': 0,
                                'confidence': 0.0,
                                'day_of_week': hour.strftime('%A'),
                                'month': hour.strftime('%B'),
                                'precipitation': (weather['precipitation'] if weather 
                                               else Precipitation.NONE.value),
                                'temperature': (weather['temperature'] if weather 
                                             else Temperature.TEN_TO_15.value),
                                'hour': hour.hour,
                                'universities_in_session': is_university_in_session(hour),
                                'ttc_closure': TTCClosure.NONE.value
                            }
                        
                        enriched_cluster.append(station_info)
                    except Exception as e:
                        print(f"Error processing station {station_name}: {e}")
                        continue
                else:
                    print(f"Warning: No coordinates found for station {station_name}")
            
            enriched_clusters[cluster_id] = enriched_cluster
        
        # Save updated weather cache
        weather_service.save_cache()
        return enriched_clusters
        
    except Exception as e:
        print(f"Error in load_clusters_with_coordinates: {e}")
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

def get_station_predictions(station: Dict[str, any], hour: datetime) -> Dict[str, any]:
    """
    Get the predicted demand and associated information for a specific station and hour.
    To be implemented with actual prediction logic and weather API.
    """
    return {
        'predicted_demand': 0,
        'confidence': 0.0,
        'day_of_week': hour.strftime('%A'),
        'month': hour.strftime('%B'),
        'precipitation': Precipitation.NONE.value,
        'temperature': Temperature.TEN_TO_15.value,
        'hour': hour.hour,
        'universities_in_session': is_university_in_session(hour)
    }

def main():
    # Load clusters with coordinates and initialize predictions
    clusters = load_clusters_with_coordinates()
    
    # Print summary of loaded data
    print_cluster_summary(clusters)
    
    # Example of accessing predictions for a specific station and time
    current_time = datetime.now()
    sample_hour = current_time + timedelta(hours=3)
    
    for cluster_id, stations in clusters.items():
        print(f"\nProcessing Cluster {cluster_id}")
        for station in stations[:1]:  # Process first station as example
            print(f"\nStation: {station['name']}")
            print(f"Predictions for {sample_hour.strftime('%Y-%m-%d %H:00')}:")
            prediction = station['predictions'][sample_hour.strftime('%Y-%m-%d %H:00:00')]
            print(f"Predicted Demand: {prediction['predicted_demand']}")
            print(f"Confidence: {prediction['confidence']:.2f}")
            print(f"Day: {prediction['day_of_week']}")
            print(f"Month: {prediction['month']}")
            print(f"Precipitation: {prediction['precipitation']}")
            print(f"Temperature: {prediction['temperature']}")

if __name__ == "__main__":
    main()

