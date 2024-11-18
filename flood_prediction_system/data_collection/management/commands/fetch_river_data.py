from django.core.management.base import BaseCommand
from django.utils import timezone
from datetime import datetime
import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
from data_collection.models import RiverData

class Command(BaseCommand):
    help = 'Fetch and save river discharge data from Open-Meteo API'

    def handle(self, *args, **kwargs):
        # Import the RiverData model here, after the apps are fully loaded

        # Setup Open-Meteo client
        cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        openmeteo = openmeteo_requests.Client(session=retry_session)

        # API call parameters
        url = "https://flood-api.open-meteo.com/v1/flood"
        params = {
            "latitude": 13.27443,
            "longitude": 121.259722,
            "daily": [
                "river_discharge", "river_discharge_mean", "river_discharge_median", 
                "river_discharge_max", "river_discharge_min", 
                "river_discharge_p25", "river_discharge_p75"
            ],
            "start_date": "2024-10-29",  
            "end_date": datetime.now().strftime('%Y-%m-%d')
        }

        # Fetch data from Open-Meteo
        try:
            response = openmeteo.weather_api(url, params=params)[0]
        except Exception as e:
            self.stderr.write(f"Failed to fetch data: {e}")
            return

        # Process daily river data
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

        self.stdout.write(self.style.SUCCESS('River data imported successfully'))
