from django.core.management.base import BaseCommand
from data_collection.models import WeatherData
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTMV1, Dense
import matplotlib.pyplot as plt

class Command(BaseCommand):
    help = 'Forecast atmospheric pressure for the next three days using LSTMV1 model'

    def handle(self, *args, **kwargs):
        # Fetch and preprocess pressure data
        self.stdout.write("Fetching pressure data...")
        weather_data = WeatherData.objects.all().values('datetime', 'pressure')
        weather_df = pd.DataFrame(weather_data)
        weather_df.set_index('datetime', inplace=True)
        weather_df.index = pd.to_datetime(weather_df.index)
        weather_df.sort_index(inplace=True)
        weather_df = weather_df[~weather_df.index.duplicated(keep='first')]

        # Normalize data
        scaler = tf.keras.layers.Normalization()
        scaler.adapt(np.array(weather_df[['pressure']]))
        normalized_data = scaler(np.array(weather_df[['pressure']]))

        # Prepare data sequences for LSTMV1
        def create_sequences(data, seq_length):
            sequences = []
            labels = []
            for i in range(len(data) - seq_length):
                sequences.append(data[i:i + seq_length])
                labels.append(data[i + seq_length])
            return np.array(sequences), np.array(labels)

        seq_length = 7
        X, y = create_sequences(normalized_data, seq_length)

        # Reshape X to be 3D (samples, time steps, features)
        X = np.reshape(X, (X.shape[0], seq_length, 1))

        # Define and train the LSTMV1 model
        model = Sequential([
            LSTMV1(64, activation='relu', input_shape=(seq_length, 1)),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=20, verbose=1)

        # Use the last sequence of data to forecast the next three days
        last_sequence = normalized_data[-seq_length:]
        predictions = []
        for _ in range(3):
            pred = model.predict(np.expand_dims(last_sequence, axis=0))
            predictions.append(pred[0, 0])
            last_sequence = np.append(last_sequence[1:], pred)
            last_sequence = np.reshape(last_sequence, (seq_length, 1))

        # Unscale predictions
        predictions_unscaled = scaler.mean.numpy() + np.array(predictions) * scaler.variance.numpy() ** 0.5

        # Print the last 10 days of pressure data
        self.stdout.write("\nLast 10 Days of Atmospheric Pressure:")
        last_10_days = weather_df.tail(10)
        for date, pressure in zip(last_10_days.index, last_10_days['pressure']):
            self.stdout.write(f"{date.strftime('%Y-%m-%d')}: {pressure:.2f} hPa")

        # Print the 3-day forecast
        self.stdout.write("\n3-Day Pressure Forecast:")
        last_date = last_10_days.index[-1]
        forecast_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 4)]
        for date, forecast in zip(forecast_dates, predictions_unscaled.flatten()):
            self.stdout.write(f"{date.strftime('%Y-%m-%d')}: {forecast:.2f} hPa")

        # Plot the historical data and forecasts
        self.plot_forecast(weather_df, predictions_unscaled)

    def plot_forecast(self, weather_df, forecast):
        plt.figure(figsize=(10, 6))

        # Get the last 10 days of historical data
        last_days = weather_df.tail(10)

        # Plot historical data for the last 10 days
        plt.plot(last_days.index, last_days['pressure'], label="Historical Pressure", color='blue')

        # Define the dates for the forecast
        last_date = last_days.index[-1]
        forecast_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 4)]

        # Flatten forecast to match forecast_dates
        forecast = forecast.flatten()

        # Plot forecasted values
        plt.plot(forecast_dates, forecast, label="3-Day Pressure Forecast", color='red', marker='o')

        # Chart formatting
        plt.xlabel("Date")
        plt.ylabel("Atmospheric Pressure (hPa)")
        plt.title("Last 10 Days of Pressure and 3-Day Forecast")
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()
