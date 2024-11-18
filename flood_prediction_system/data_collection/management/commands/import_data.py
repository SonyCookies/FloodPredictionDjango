# file_path = 'data_collection/data/Naujan 2022-01-01 to 2024-09-25.xlsx'  
# import_data.py

from django.core.management.base import BaseCommand
import json
from data_collection.models import WeatherData
from django.utils import timezone
from datetime import datetime

class Command(BaseCommand):
    help = 'Import weather data from JSON'

    def handle(self, *args, **kwargs):
        file_path = 'data_collection/data/Naujan 2024-09-26 to 2024-10-23.txt'  

        with open(file_path, 'r') as file:
            data = json.load(file)

        for day in data['days']:
            datetime_value = timezone.make_aware(datetime.strptime(day['datetime'], '%Y-%m-%d'))
            preciptype = day.get('preciptype', None)
            if isinstance(preciptype, list):
                preciptype = ', '.join(preciptype)
            else:
                preciptype = str(preciptype) if preciptype is not None else ''

            WeatherData.objects.create(
                name=data['name'],
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
                preciptype=preciptype,  # Use the modified preciptype
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

        self.stdout.write(self.style.SUCCESS('Data imported successfully'))
