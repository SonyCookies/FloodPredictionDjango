from django.core.management.base import BaseCommand
from data_collection.models import RiverData
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTMV1, Dense
import matplotlib.pyplot as plt

class Command(BaseCommand):
    help = 'Forecast river discharge for the next three days using LSTM model'

    def handle(self, *args, **kwargs):
        # Fetch and preprocess ng river data
        self.stdout.write("Fetching river data...")
        river_data = RiverData.objects.all().values('date', 'river_discharge')
        river_df = pd.DataFrame(river_data)
        river_df.set_index('date', inplace=True)
        river_df.index = pd.to_datetime(river_df.index)
        river_df.sort_index(inplace=True)
        river_df = river_df[~river_df.index.duplicated(keep='first')]

        # Pag Normalize ng data
        scaler = tf.keras.layers.Normalization()
        scaler.adapt(np.array(river_df[['river_discharge']]))
        normalized_data = scaler(np.array(river_df[['river_discharge']]))

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
            last_sequence = np.reshape(last_sequence, (seq_length, 1))  # Ensure shape is compatible

        # Unscale predictions
        predictions_unscaled = scaler.mean.numpy() + np.array(predictions) * scaler.variance.numpy() ** 0.5

        # Display the last 10 days of historical data and the 3-day forecast
        self.display_forecast(river_df, predictions_unscaled)

        # Plot the historical data and forecasts
        self.plot_forecast(river_df, predictions_unscaled)

    def display_forecast(self, river_df, forecast):
        # Get the last 10 days of historical data
        last_days = river_df.tail(10)

        # Print the last 10 days of river discharge data
        print("Last 10 Days of River Discharge:")
        for date, discharge in last_days['river_discharge'].items():
            print(f"{date.date()}: {discharge}")

        # Define the dates for the 3-day forecast
        last_date = last_days.index[-1]
        forecast_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 4)]

        # Print the forecasted values, one per line with each date
        print("\n3-Day Forecast:")
        for date, value in zip(forecast_dates, forecast.flatten()):  # Use flatten() to unpack forecast
            print(f"{date.date()}: {value}")


    def plot_forecast(self, river_df, forecast):
        plt.figure(figsize=(10, 6))

        last_days = river_df.tail(10)

        # Plot historical data for the last 10 days
        plt.plot(last_days.index, last_days['river_discharge'], label="Historical River Discharge", color='blue')

        # Define the dates for the forecast
        last_date = last_days.index[-1]
        forecast_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 4)]

        # Flatten forecast to match forecast_dates
        forecast = forecast.flatten()

        # Plot forecasted values
        plt.plot(forecast_dates, forecast, label="3-Day Forecast", color='red', marker='o')

        # Chart formatting
        plt.xlabel("Date")
        plt.ylabel("River Discharge")
        plt.title("Last 10 Days of River Discharge and 3-Day Forecast")
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
        plt.legend()
        plt.grid()
        plt.tight_layout()  # Adjust layout to make room for the rotated labels
        plt.show()
