# fetch_weather_data.py (inside your management/commands folder)

import requests
from django.core.management.base import BaseCommand
from data_collection.models import WeatherData
from django.utils import timezone
from datetime import datetime
from django.conf import settings  # Import settings to access the API key



class Command(BaseCommand):
    help = 'Fetch weather data from Visual Crossing API and save it to the database'

    def handle(self, *args, **kwargs):
        # Define your Visual Crossing API URL and parameters
        api_key = settings.VISUAL_CROSSING_API_KEY 

        if not api_key:
            self.stdout.write(self.style.ERROR('API key is missing.'))
            return
        
        location = 'Naujan'
        base_url = f'https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{location}/2024-10-29/2024-11-4'
        params = {
            'unitGroup': 'metric',   # You can change to 'us' if you prefer imperial units
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

                # Save each day's weather data to the database
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
                    stations=', '.join(day.get('stations', []))  # Join list to string
                )

            self.stdout.write(self.style.SUCCESS('Weather data fetched and saved successfully'))

        else:
            self.stdout.write(self.style.ERROR(f'Failed to fetch weather data. Status code: {response.status_code}'))
