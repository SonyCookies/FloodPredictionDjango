from django.core.management.base import BaseCommand
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from data_collection.models import RiverData

class Command(BaseCommand):
    help = 'Check the stationarity of river discharge data'

    def handle(self, *args, **kwargs):
        # Load river data from the database
        queryset = RiverData.objects.all().values('date', 'river_discharge')
        data = pd.DataFrame(list(queryset))

        if data.empty:
            print("No data found in the database.")
            return

        # Preprocessing
        data['date'] = pd.to_datetime(data['date'])
        data.set_index('date', inplace=True)

        # Differencing to make the series stationary
        data['river_discharge_diff'] = data['river_discharge'].diff()
        data.dropna(inplace=True)

        # ADF Test
        adf_result = adfuller(data['river_discharge_diff'])
        print(f'ADF Statistic: {adf_result[0]}')
        print(f'p-value: {adf_result[1]}')
        print('Critical Values:')
        for key, value in adf_result[4].items():
            print(f'   {key}: {value}')

        # Print result
        if adf_result[1] < 0.05:
            print("Reject the null hypothesis - The time series is stationary.")
        else:
            print("Fail to reject the null hypothesis - The time series is non-stationary.")

        print("Stationarity check completed.")
