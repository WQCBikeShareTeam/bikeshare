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

def load_historical_data():
    """Load and combine historical trip data"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_file = os.path.join(base_dir, "Region_Creation_Parsing", "Bike share ridership 2023-07.csv")
    
    try:
        df = pd.read_csv(data_file)
        # Convert timestamps to datetime
        df['Start Time'] = pd.to_datetime(df['Start Time'])
        df['End Time'] = pd.to_datetime(df['End Time'])
        return df
    except FileNotFoundError:
        print(f"Warning: Could not find data file: {data_file}")
        raise

def load_clusters_with_coordinates():
    """Load clusters and enrich them with station coordinates"""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        clusters_path = os.path.join(current_dir, "..", "Region_Creation_Parsing", "station_clusters.json")
        coordinates_path = os.path.join(current_dir, "..", "bikeshare-api", "get_clusters", "station_coordinates_cache.json")
        
        # Load files
        with open(clusters_path, 'r') as f:
            clusters_data = json.load(f)
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

def prepare_station_data(historical_data: pd.DataFrame, station_name: str) -> pd.DataFrame:
    """
    Prepare features for a specific station from historical data
    """
    # Convert timestamps to datetime
    historical_data['Start Time'] = pd.to_datetime(historical_data['Start Time'])
    historical_data['End Time'] = pd.to_datetime(historical_data['End Time'])
    
    # Create outflow data (bikes leaving the station)
    outflow = historical_data[historical_data['Start Station Name'] == station_name].copy()
    outflow['Hour'] = outflow['Start Time'].dt.hour
    outflow['DayOfWeek'] = outflow['Start Time'].dt.day_name()
    outflow['Temperature'] = outflow['Temp (°C)']
    outflow['Precipitation'] = outflow['Precip. Amount (mm)']
    outflow['flow'] = -1  # Negative for outflow
    
    # Create inflow data (bikes arriving at the station)
    inflow = historical_data[historical_data['End Station Name'] == station_name].copy()
    inflow['Hour'] = inflow['End Time'].dt.hour
    inflow['DayOfWeek'] = inflow['End Time'].dt.day_name()
    inflow['Temperature'] = inflow['Temp (°C)']
    inflow['Precipitation'] = inflow['Precip. Amount (mm)']
    inflow['flow'] = 1  # Positive for inflow
    
    # Combine flows
    flows = pd.concat([outflow, inflow])
    
    # Group by hour and features, sum flows
    hourly_flows = flows.groupby([
        'Hour', 
        'DayOfWeek', 
        'Temperature',
        'Precipitation'
    ])['flow'].sum().reset_index()
    
    return hourly_flows

def train_prediction_model(station_name: str) -> Tuple[RandomForestRegressor, LabelEncoder, Dict]:
    """
    Train a prediction model for a specific station
    """
    # Load historical data (August only for now)
    historical_data = load_historical_data()
    historical_data = historical_data[historical_data['Start Time'].dt.month == 8]
    
    # Prepare features
    station_data = prepare_station_data(historical_data, station_name)
    
    # Encode categorical variables
    le_day = LabelEncoder()
    station_data['DayOfWeek_encoded'] = le_day.fit_transform(station_data['DayOfWeek'])
    
    # Create temperature and precipitation categories
    def temp_to_category(temp):
        if temp < -10: return "below -10°C"
        elif temp < 0: return "-5°C to 0°C"
        elif temp < 5: return "0°C to 5°C"
        elif temp < 10: return "5°C to 10°C"
        elif temp < 15: return "10°C to 15°C"
        elif temp < 20: return "15°C to 20°C"
        elif temp < 25: return "20°C to 25°C"
        else: return "above 25°C"
    
    def precip_to_category(precip):
        if precip == 0: return "none"
        elif precip < 2.5: return "light rain"
        elif precip < 7.5: return "heavy rain"
        else: return "heavy rain"
    
    station_data['temp_category'] = station_data['Temperature'].apply(temp_to_category)
    station_data['precip_category'] = station_data['Precipitation'].apply(precip_to_category)
    
    le_temp = LabelEncoder()
    le_precip = LabelEncoder()
    station_data['temp_encoded'] = le_temp.fit_transform(station_data['temp_category'])
    station_data['precip_encoded'] = le_precip.fit_transform(station_data['precip_category'])
    
    # Prepare features and target
    X = station_data[[
        'Hour', 
        'DayOfWeek_encoded',
        'temp_encoded',
        'precip_encoded'
    ]]
    y = station_data['flow']
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    encoders = {
        'day': le_day,
        'temp': le_temp,
        'precip': le_precip
    }
    
    return model, encoders

def get_station_predictions(station: Dict[str, any], hour: datetime) -> List[Dict[str, any]]:
    """Get predictions for a station for the next 24 hours"""
    try:
        # Train model for this station
        model, encoders = train_prediction_model(station['name'])
        
        predictions = []
        current_bikes = 25  # Starting assumption
        ##Generate real amount of bikes
        # Generate predictions for next 24 hours
        for i in range(24):
            prediction_time = hour + timedelta(hours=i)
            prediction_key = prediction_time.strftime('%Y-%m-%d %H:00:00')
            
            # Get weather for this hour
            weather_pred = station['predictions'][prediction_key]
            
            # Prepare features for prediction
            features = pd.DataFrame({
                'Hour': [prediction_time.hour],
                'DayOfWeek_encoded': [encoders['day'].transform([prediction_time.strftime('%A')])[0]],
                'temp_encoded': [encoders['temp'].transform([weather_pred['temperature']])[0]],
                'precip_encoded': [encoders['precip'].transform([weather_pred['precipitation']])[0]]
            })
            
            # Make prediction
            predicted_flow = model.predict(features)[0]
            
            # Update bike count
            new_bikes = max(0, min(35, current_bikes + predicted_flow))
            #####This Needs to be changed too the actual bike count at the sttaion
            ## We need to put the minimum and maximum amount of bikes at the station
            predictions.append({
                'hour': prediction_key,
                'net_flow': round(predicted_flow, 2),
                'predicted_bikes': round(new_bikes),
                'confidence': 0.8,  # TODO: Implement proper confidence calculation
                'day_of_week': prediction_time.strftime('%A'),
                'temperature': weather_pred['temperature'],
                'precipitation': weather_pred['precipitation']
            })
            
            current_bikes = new_bikes
            ## We need a new function to determine payout of each station
            ## For given hour, for a given station in the next 24 hours
                    # If (predicted bike amount is greter then capacity, trigger calculate payout function). Take the difference beteween predicted bikes
                    # and the capacity
                    # DO the same if the predicted bikeS LESS the zero (Take difference)
            
            ## for payout function
            ##IF less then zero, for that particular station (AADD new columns too station)
            ##Add Column, Ride Time
            ##Add Column, Is Electric Bike 
            ##Add Column, Is Premium Member
            ##Calculate total payout for that ride
            ##Find the costs for bikes per hour and shit.
            ##Calculate the income from rider based on if electric or if regular bike.
            ##If premium membe, calculate as if casual rider but multiply total by 0.8 (cause they give less money)
            ## Add these columns (including the payout columns) to the main data frame.
            ##Now using predictions calculate net payout aand include it in the station predictions.
            ##Create a new function that for each station, it takes their stataion predictions and finds the cost for that station
            ##Then we will look at clusters and weigh cost (See setuptruckroute optimiataion too see how it flows in.) 
                       
        return predictions
        
    except Exception as e:
        print(f"Error predicting for station {station['name']}: {e}")
        return []

def print_station_predictions(predictions: List[Dict[str, any]]):
    """Print predictions in a formatted table"""
    print("\nHourly Predictions:")
    print("-" * 120)
    print(f"{'Hour':<20} {'Net Flow':>10} {'Predicted Bikes':>15} {'Temperature':>15} {'Precipitation':>15}")
    print("-" * 120)
    
    for pred in predictions:
        print(f"{pred['hour']:<20} {pred['net_flow']:>10.2f} {pred['predicted_bikes']:>15.0f} "
              f"{pred['temperature']:>15} {pred['precipitation']:>15}")

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
        for station in stations[:3]:  # Process first station as example
            print(f"\nStation: {station['name']}")
            print(f"Predictions for {sample_hour.strftime('%Y-%m-%d %H:00')}:")
            predictions = get_station_predictions(station, sample_hour)
            print_station_predictions(predictions)
    stations = ['Station A', 'Station B', 'Station C', 'Station D']
    
if __name__ == "__main__":
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

