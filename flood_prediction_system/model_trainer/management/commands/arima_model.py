from django.core.management.base import BaseCommand
from data_collection.models import WeatherData, RiverData
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

class Command(BaseCommand):
    help = 'Build ARIMA model for weather and river flood prediction'

    def handle(self, *args, **kwargs):
        # Fetch weather data for weather prediction
        self.stdout.write("Fetching weather data...")
        weather_data = WeatherData.objects.all().values('datetime', 'temp')
        weather_df = pd.DataFrame(weather_data)
        weather_df.set_index('datetime', inplace=True)

        if weather_df.empty:
            print("No weather data found.")
            return

        # Check stationarity of weather data
        print("Checking stationarity for weather data...")
        weather_df['temp'], weather_diff = self.check_stationarity(weather_df['temp'])

        # Build ARIMA model for weather data
        print("Building ARIMA model for weather data...")
        arima_model_weather = ARIMA(weather_diff, order=(1, 1, 1))  
        weather_results = arima_model_weather.fit()
        print(weather_results.summary())

        # Make weather predictions
        forecast_weather = weather_results.forecast(steps=5)
        print("Weather forecast for next 30 days:", forecast_weather)

    # Fetch river data for flood prediction
        self.stdout.write("Fetching river data...")
        river_data = RiverData.objects.all().values('date', 'river_discharge')
        river_df = pd.DataFrame(river_data)
        
        # Ensure the index is unique by removing duplicates or handling them
        river_df.set_index('date', inplace=True)
        
        if river_df.index.duplicated().any():
            print("Duplicate dates found in river data. Handling duplicates...")
            # Option 1: Remove duplicates, keeping the first occurrence
            river_df = river_df[~river_df.index.duplicated(keep='first')]
            
            # Option 2: If you'd rather average the values for duplicate dates, use this:
            # river_df = river_df.groupby('date').mean()

        if river_df.empty:
            print("No river data found.")
            return

        # Check stationarity of river data
        print("Checking stationarity for river data...")
        river_df['river_discharge'], river_diff = self.check_stationarity(river_df['river_discharge'])

        # Build ARIMA model for river data
        print("Building ARIMA model for river data...")
        arima_model_river = ARIMA(river_diff, order=(1, 1, 1))  # Adjust order based on checks
        river_results = arima_model_river.fit()
        print(river_results.summary())

        # Last observed value of river discharge before forecasting
        last_observed_value = river_df['river_discharge'].iloc[-1]

        # Reverse differencing by adding back the last observed value
        forecast_river = river_results.forecast(steps=5)  # This gives differenced predictions
        reverse_forecast = forecast_river.cumsum() + last_observed_value

        print("Reverse differenced river discharge forecast:", reverse_forecast)

        # Optionally, plot results
        self.plot_results(forecast_weather, reverse_forecast)

    def check_stationarity(self, timeseries):
        # Perform Augmented Dickey-Fuller test
        result = adfuller(timeseries.dropna())  # Drop NaNs to avoid errors
        print('ADF Statistic:', result[0])
        print('p-value:', result[1])
        print('Critical Values:')
        for key, value in result[4].items():
            print(f'   {key}: {value}')

        # Check if the series is stationary or not
        if result[1] > 0.05:
            print("Fail to reject the null hypothesis - The time series is non-stationary.")
            # Apply differencing
            print("Applying differencing...")
            differenced_series = timeseries.diff().dropna()
            # Re-check stationarity after differencing
            return differenced_series, differenced_series
        else:
            print("Reject the null hypothesis - The time series is stationary.")
            return timeseries, timeseries

    def plot_results(self, weather_forecast, river_forecast):
        plt.figure(figsize=(14, 6))

        plt.subplot(1, 2, 1)
        plt.plot(weather_forecast, label="Weather Forecast")
        plt.title('Weather Forecast')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(river_forecast, label="River Flood Forecast")
        plt.title('Flood Forecast')
        plt.legend()

        plt.tight_layout()
        plt.show()
