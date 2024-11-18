from django.core.management.base import BaseCommand
from data_collection.models import WeatherData
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTMV1, Dense
import matplotlib.pyplot as plt

class Command(BaseCommand):
    help = 'Forecast temperature for the next three days using LSTMV1 model'

    def handle(self, *args, **kwargs):
        # Fetch and preprocess temperature data
        self.stdout.write("Fetching temperature data...")
        weather_data = WeatherData.objects.all().values('datetime', 'temp')
        weather_df = pd.DataFrame(weather_data)
        weather_df.set_index('datetime', inplace=True)
        weather_df.index = pd.to_datetime(weather_df.index)
        weather_df.sort_index(inplace=True)
        weather_df = weather_df[~weather_df.index.duplicated(keep='first')]

        # Normalize data
        scaler = tf.keras.layers.Normalization()
        scaler.adapt(np.array(weather_df[['temp']]))
        normalized_data = scaler(np.array(weather_df[['temp']]))

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

        # Print the last 10 days of temperature data
        self.stdout.write("\nLast 10 Days of Temperature:")
        last_10_days = weather_df.tail(10)
        for date, temp in zip(last_10_days.index, last_10_days['temp']):
            self.stdout.write(f"{date.strftime('%Y-%m-%d')}: {temp:.2f}°C")

        # Print the 3-day forecast
        self.stdout.write("\n3-Day Temperature Forecast:")
        last_date = last_10_days.index[-1]
        forecast_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 4)]
        for date, forecast in zip(forecast_dates, predictions_unscaled.flatten()):
            self.stdout.write(f"{date.strftime('%Y-%m-%d')}: {forecast:.2f}°C")

        # Plot the historical data and forecasts
        self.plot_forecast(weather_df, predictions_unscaled)

    def plot_forecast(self, weather_df, forecast):
        plt.figure(figsize=(10, 6))

        # Get the last 10 days of historical data
        last_days = weather_df.tail(10)

        # Plot historical data for the last 10 days
        plt.plot(last_days.index, last_days['temp'], label="Historical Temperature", color='blue')

        # Define the dates for the forecast
        last_date = last_days.index[-1]
        forecast_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 4)]

        # Flatten forecast to match forecast_dates
        forecast = forecast.flatten()

        # Plot forecasted values
        plt.plot(forecast_dates, forecast, label="3-Day Temperature Forecast", color='red', marker='o')

        # Chart formatting
        plt.xlabel("Date")
        plt.ylabel("Temperature")
        plt.title("Last 10 Days of Temperature and 3-Day Forecast")
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()
