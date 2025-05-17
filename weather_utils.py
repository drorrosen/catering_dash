import requests
import pandas as pd
from datetime import date, timedelta

def get_weather_data(start_date_dt: date, end_date_dt: date):
    """
    Fetches and processes weather data from the Open-Meteo API for Redmond, WA for a given date range.

    Args:
        start_date_dt (date): The start date for the weather data.
        end_date_dt (date): The end date for the weather data.
    """
    # Coordinates for Redmond, WA
    latitude = 47.6739881
    longitude = -122.121512

    # API URL and parameters
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date_dt.isoformat(),
        "end_date": end_date_dt.isoformat(),
        "daily": ",".join([
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "rain_sum",
            "snowfall_sum",
            "windspeed_10m_max",
            "weathercode",
            "sunrise",
            "sunset"
        ]),
        "timezone": "America/Los_Angeles"
    }

    try:
        # Request data from API
        response = requests.get(url, params=params)
        response.raise_for_status() # Raise an exception for HTTP errors
        data = response.json()

        # Convert to DataFrame
        df = pd.DataFrame(data['daily'])
        
        # Convert date column to datetime.date objects
        df['date'] = pd.to_datetime(df['time']).dt.date 
        df = df.drop(columns=['time']) # Drop original time column

        df['sunrise'] = pd.to_datetime(df['sunrise'])
        df['sunset'] = pd.to_datetime(df['sunset'])
        df['daylight_time'] = df['sunset'] - df['sunrise']

        # Convert the 'daylight_time' column to total seconds
        df['daylight_time_seconds'] = df['daylight_time'].dt.total_seconds()
        # Convert seconds to hours (decimal)
        df['daylight_time_hours'] = df['daylight_time_seconds'] / 3600
        # Drop the original 'daylight_time' and the intermediate 'daylight_time_seconds' column
        df = df.drop(columns=['daylight_time', 'daylight_time_seconds'])
        # Rename the new column to 'daylight_time' for consistency
        df = df.rename(columns={'daylight_time_hours': 'daylight_time'})

        #drop the sunrise and sunset colums
        df = df.drop(columns=['sunrise', 'sunset'])
        
        return df
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return pd.DataFrame() # Return empty DataFrame on error
    except KeyError as e:
        print(f"Error processing weather data (unexpected format): {e}")
        return pd.DataFrame() # Return empty DataFrame on error 