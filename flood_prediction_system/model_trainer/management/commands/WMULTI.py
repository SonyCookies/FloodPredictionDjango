from django.core.management.base import BaseCommand
from data_collection.models import WeatherData
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

class Command(BaseCommand):
    help = 'Multivariate LSTM forecast using humidity, temperature, dew, and pressure'

    def handle(self, *args, **kwargs):
        self.stdout.write("Fetching weather data...")
        
        # Load weather data
        data = WeatherData.objects.all().values('datetime', 'humidity', 'temp', 'dew', 'pressure')
        df = pd.DataFrame(data)
        df.set_index('datetime', inplace=True)
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        df = df[~df.index.duplicated(keep='first')]

        # Normalize each feature
        scaler = tf.keras.layers.Normalization()
        scaler.adapt(np.array(df[['humidity', 'temp', 'dew', 'pressure']]))
        normalized_data = scaler(np.array(df[['humidity', 'temp', 'dew', 'pressure']]))

        # Sequence preparation for LSTM (7 steps in history to predict next step)
        seq_length = 7

        def create_sequences(data, seq_length):
            sequences = []
            labels = []
            for i in range(len(data) - seq_length):
                sequences.append(data[i:i + seq_length])
                labels.append(data[i + seq_length, 1])  # Example: predict 'temp' as target
            return np.array(sequences), np.array(labels)

        X, y = create_sequences(normalized_data, seq_length)
        
        # Reshape X to 3D (samples, time steps, features)
        X = np.reshape(X, (X.shape[0], seq_length, 4))  # 4 features in each timestep

        # Define the LSTM model
        model = Sequential([
            LSTM(64, activation='relu', input_shape=(seq_length, 4)),
            Dense(1)  # Assuming predicting temperature
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=20, verbose=1)

        # Forecasting
        last_sequence = normalized_data[-seq_length:]  # Last sequence of recent data
        predictions = []

        for _ in range(3):  # Forecast next 3 time steps
            pred = model.predict(np.expand_dims(last_sequence, axis=0))
            predictions.append(pred[0, 0])
            last_sequence = np.append(last_sequence[1:], np.expand_dims(pred, axis=1), axis=0)

        # Unscale predictions for readability
        predictions_unscaled = scaler.mean.numpy()[1] + np.array(predictions) * scaler.variance.numpy()[1] ** 0.5

        self.stdout.write("\n3-Day Forecast (Temperature):")
        forecast_dates = [df.index[-1] + pd.Timedelta(days=i) for i in range(1, 4)]
        for date, forecast in zip(forecast_dates, predictions_unscaled.flatten()):
            self.stdout.write(f"{date.strftime('%Y-%m-%d')}: {forecast:.2f}°C")
