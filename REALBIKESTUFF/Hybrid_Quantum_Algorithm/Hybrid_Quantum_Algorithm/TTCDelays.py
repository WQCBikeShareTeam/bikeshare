import requests

from datetime import datetime, timedelta

# Define the API endpoint and parameters
API_URL = "https://alerts.ttc.ca/api/alerts/list"
# Calculate the current time and 6 hours ahead in ISO 8601 format
current_time = datetime.utcnow()
end_time = current_time + timedelta(hours=6)

# Convert to strings
current_time_str = current_time.strftime("%Y-%m-%dT%H:%M:%SZ")
end_time_str = end_time.strftime("%Y-%m-%dT%H:%M:%SZ")

# Example payload (if the API supports time filtering)
params = {
    "start_time": current_time_str,  # Optional parameter depending on the API
    "end_time": end_time_str,       # Optional parameter depending on the API
}

# Make the API request
response = requests.get(API_URL, params=params)

# Check for success
if response.status_code == 200:
    alerts = response.json()
    print("Service Alerts:")
    for alert in alerts.get("alerts", []):
        print(f"Alert ID: {alert['alert_id']}")
        print(f"Description: {alert['description']}")
        print(f"Start Time: {alert['start_time']}")
        print(f"End Time: {alert['end_time']}")
        print("-" * 40)
else:
    print(f"Error: Unable to fetch data. Status code: {response.status_code}")
