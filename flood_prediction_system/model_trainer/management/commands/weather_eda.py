from django.core.management.base import BaseCommand
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from data_collection.models import WeatherData
import numpy as np
import calendar

class Command(BaseCommand):
    help = 'Preprocess and analyze weather data for modeling'

    def handle(self, *args, **kwargs):
        # Load data from the database
        queryset = WeatherData.objects.all().values(
            'datetime', 'tempmax', 'tempmin', 'temp', 'feelslikemax', 'feelslikemin', 'feelslike',
            'dew', 'humidity', 'precip', 'precipprob', 'precipcover', 'preciptype', 'snow', 'snowdepth',
            'windgust', 'windspeed', 'winddir', 'pressure', 'cloudcover', 'visibility', 'solarradiation',
            'solarenergy', 'uvindex', 'sunrise', 'sunset', 'moonphase', 'conditions', 'description', 
            'icon', 'stations'
        )
        
        # Convert queryset to DataFrame
        data = pd.DataFrame(list(queryset))

        if data.empty:
            print("No data found in the database.")
            return

        # Preprocessing
        data['datetime'] = pd.to_datetime(data['datetime'])
        data['precip_lag1'] = data['precip'].shift(1)
        data.set_index('datetime', inplace=True)

        # EDA Part - Exploratory Data Analysis
        # 1. Summary statistics
        print("\nSummary statistics:")
        print(data.describe())

        # 2. Data distribution: Temperature, Humidity, and Precipitation
        plt.figure(figsize=(14, 6))
        sns.histplot(data['temp'], bins=30, kde=True, color='blue')
        plt.title('Distribution of Temperature')
        plt.show()

        plt.figure(figsize=(14, 6))
        sns.histplot(data['humidity'], bins=30, kde=True, color='green')
        plt.title('Distribution of Humidity')
        plt.show()

        plt.figure(figsize=(14, 6))
        sns.histplot(data['precip'], bins=30, kde=True, color='purple')
        plt.title('Distribution of Precipitation')
        plt.show()

        # 3. Correlation heatmap
        plt.figure(figsize=(12, 8))
        corr_matrix = data[['tempmax', 'tempmin', 'temp', 'feelslikemax', 'humidity', 'precip', 'windspeed', 'pressure']].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Heatmap')
        plt.show()

        # 4. Time series decomposition for precipitation
        decomposition = seasonal_decompose(data['precip'].dropna(), model='additive', period=30)  # Assuming monthly seasonality
        decomposition.plot()
        plt.show()

        # 5. Line plot for temperature and precipitation over time
        plt.figure(figsize=(14, 6))
        plt.plot(data.index, data['temp'], label='Temperature', color='blue', alpha=0.7)
        plt.plot(data.index, data['precip'], label='Precipitation', color='purple', alpha=0.7)
        plt.legend()
        plt.title('Temperature and Precipitation Over Time')
        plt.show()

        # 6. Box plots for monthly temperature and precipitation
        data['month'] = data.index.month
        plt.figure(figsize=(14, 6))
        sns.boxplot(x=data['month'], y=data['temp'], palette='Blues')
        plt.title('Monthly Temperature Distribution')
        plt.xticks(np.arange(12), calendar.month_name[1:13], rotation=45)
        plt.show()

        plt.figure(figsize=(14, 6))
        sns.boxplot(x=data['month'], y=data['precip'], palette='Purples')
        plt.title('Monthly Precipitation Distribution')
        plt.xticks(np.arange(12), calendar.month_name[1:13], rotation=45)
        plt.show()

        # 7. Pairplot for multiple variables
        sns.pairplot(data[['tempmax', 'tempmin', 'temp', 'humidity', 'precip', 'windspeed']].dropna())
        plt.show()

        # 8. Simple ARIMA forecasting for precipitation
        model = ARIMA(data['precip'].dropna(), order=(5, 1, 0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=30)

        plt.figure(figsize=(14, 6))
        plt.plot(data.index, data['precip'], label='Observed', color='blue')
        plt.plot(pd.date_range(start=data.index[-1], periods=30, freq='D'), forecast, label='Forecast', color='red')
        plt.title('ARIMA Forecast for Precipitation')
        plt.legend()
        plt.show()

        # 9. Windrose chart for wind direction and speed
        from windrose import WindroseAxes
        ax = WindroseAxes.from_ax()
        ax.bar(data['winddir'], data['windspeed'], normed=True, opening=0.8, edgecolor='white')
        ax.set_legend()
        plt.title('Windrose: Wind Direction and Speed')
        plt.show()

        print("Exploratory Data Analysis completed.")
