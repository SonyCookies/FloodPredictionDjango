from .models import RiverData, WeatherData  # Import your models
import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
from django.utils import timezone
from datetime import datetime
from django.conf import settings
import requests

def fetch_river_data():
    try:
        cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        openmeteo = openmeteo_requests.Client(session=retry_session)
        today = datetime.now().strftime('%Y-%m-%d')

        url = "https://flood-api.open-meteo.com/v1/flood"
        params = {
            "latitude": 13.27443,
            "longitude": 121.259722,
            "daily": ["river_discharge", "river_discharge_mean", "river_discharge_median", "river_discharge_max", "river_discharge_min", "river_discharge_p25", "river_discharge_p75"],
            "start_date": today,  # Fetch today's data
            "end_date": today
        }
        response = openmeteo.weather_api(url, params=params)[0]

        daily = response.Daily()
        daily_data = {
            "date": pd.date_range(
                start=pd.to_datetime(daily.Time(), unit="s", utc=True),
                end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=daily.Interval()),
                inclusive="left"
            ),
            "river_discharge": daily.Variables(0).ValuesAsNumpy(),
            "river_discharge_mean": daily.Variables(1).ValuesAsNumpy(),
            "river_discharge_median": daily.Variables(2).ValuesAsNumpy(),
            "river_discharge_max": daily.Variables(3).ValuesAsNumpy(),
            "river_discharge_min": daily.Variables(4).ValuesAsNumpy(),
            "river_discharge_p25": daily.Variables(5).ValuesAsNumpy(),
            "river_discharge_p75": daily.Variables(6).ValuesAsNumpy(),
        }

        # Convert daily data to a DataFrame for easy iteration
        daily_dataframe = pd.DataFrame(data=daily_data)

        # Save data to the database
        for index, row in daily_dataframe.iterrows():
            RiverData.objects.create(
                date=timezone.make_aware(datetime.strptime(str(row['date'].date()), '%Y-%m-%d')),
                river_discharge=row['river_discharge'],
                river_discharge_mean=row['river_discharge_mean'],
                river_discharge_median=row['river_discharge_median'],
                river_discharge_max=row['river_discharge_max'],
                river_discharge_min=row['river_discharge_min'],
                river_discharge_p25=row['river_discharge_p25'],
                river_discharge_p75=row['river_discharge_p75']
            )

        print("River data imported successfully!")

    except Exception as e:
        print(f"Error fetching river data: {e}")

def fetch_weather_data():
    try:
        api_key = settings.VISUAL_CROSSING_API_KEY
        if not api_key:
            print("API key is missing.")
            return
        
        location = 'Naujan'
        today = datetime.now().strftime('%Y-%m-%d')
        base_url = f'https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{location}/{today}'
        params = {
            'unitGroup': 'metric',
            'key': api_key,
            'include': 'days'
        }

        response = requests.get(base_url, params=params)

        if response.status_code == 200:
            data = response.json()

            for day in data['days']:
                datetime_value = timezone.make_aware(datetime.strptime(day['datetime'], '%Y-%m-%d'))
                
                preciptype = day.get('preciptype', None)
                if isinstance(preciptype, list):
                    preciptype = ', '.join(preciptype)
                else:
                    preciptype = str(preciptype) if preciptype is not None else ''

                WeatherData.objects.create(
                    name=data['address'],
                    datetime=datetime_value,
                    tempmax=day.get('tempmax', 0),
                    tempmin=day.get('tempmin', 0),
                    temp=day.get('temp', 0),
                    feelslikemax=day.get('feelslikemax', 0),
                    feelslikemin=day.get('feelslikemin', 0),
                    feelslike=day.get('feelslike', 0),
                    dew=day.get('dew', 0),
                    humidity=day.get('humidity', 0),
                    precip=day.get('precip', 0),
                    precipprob=day.get('precipprob', 0),
                    precipcover=day.get('precipcover', 0),
                    preciptype=preciptype,
                    snow=day.get('snow', 0),
                    snowdepth=day.get('snowdepth', 0),
                    windgust=day.get('windgust', 0),
                    windspeed=day.get('windspeed', 0),
                    winddir=day.get('winddir', 0),
                    pressure=day.get('pressure', 0),
                    cloudcover=day.get('cloudcover', 0),
                    visibility=day.get('visibility', 0),
                    solarradiation=day.get('solarradiation', 0),
                    solarenergy=day.get('solarenergy', 0),
                    uvindex=day.get('uvindex', 0),
                    sunrise=day.get('sunrise', None),
                    sunset=day.get('sunset', None),
                    moonphase=day.get('moonphase', None),
                    conditions=day.get('conditions', None),
                    description=day.get('description', None),
                    icon=day.get('icon', None),
                    stations=', '.join(day.get('stations', []))
                )

            print("Weather data fetched and saved successfully")

        else:
            print(f"Failed to fetch weather data. Status code: {response.status_code}")

    except Exception as e:
        print(f"Error fetching weather data: {e}")