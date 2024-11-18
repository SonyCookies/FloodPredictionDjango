from django.core.management.base import BaseCommand
from data_collection.models import RiverData
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTMV1, Dense
import matplotlib.pyplot as plt

class Command(BaseCommand):
    help = 'Build and evaluate LSTM model for river flood prediction with fine-tuning'

    def handle(self, *args, **kwargs):
        self.stdout.write("Fetching river data...")
        river_data = RiverData.objects.all().values('date', 'river_discharge')
        river_df = pd.DataFrame(river_data)
        
        river_df.set_index('date', inplace=True)
        river_df.index = pd.to_datetime(river_df.index)
        river_df.sort_index(inplace=True)
        river_df = river_df[~river_df.index.duplicated(keep='first')]
        
        train_size = int(len(river_df) * 0.8)
        train, test = river_df[:train_size], river_df[train_size:]

        scaler = tf.keras.layers.Normalization()
        scaler.adapt(np.array(train[['river_discharge']]))
        train_normalized = scaler(np.array(train[['river_discharge']]))
        test_normalized = scaler(np.array(test[['river_discharge']]))

        def create_sequences(data, seq_length):
            sequences = []
            labels = []
            for i in range(len(data) - seq_length):
                sequences.append(data[i:i + seq_length])
                labels.append(data[i + seq_length])
            return np.array(sequences), np.array(labels)

        seq_length = 14  
        hidden_units = 64  
        epochs = 50  
        batch_size = 16  


        X_train, y_train = create_sequences(train_normalized, seq_length)
        X_test, y_test = create_sequences(test_normalized, seq_length)

        model = Sequential([
            LSTMV1(hidden_units, activation='relu', input_shape=(seq_length, 1)),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')


        self.stdout.write("Training LSTM model...")
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=1)
        

        self.stdout.write("Generating forecasts...")
        predictions = model.predict(X_test)


        predictions_unscaled = scaler.mean.numpy() + predictions * scaler.variance.numpy() ** 0.5
        y_test_unscaled = scaler.mean.numpy() + y_test * scaler.variance.numpy() ** 0.5


        mae = mean_absolute_error(y_test_unscaled, predictions_unscaled)
        mse = mean_squared_error(y_test_unscaled, predictions_unscaled)
        r2 = r2_score(y_test_unscaled, predictions_unscaled)
        
  
        self.stdout.write(f"Mean Absolute Error (MAE): {mae}")
        self.stdout.write(f"Mean Squared Error (MSE): {mse}")
        self.stdout.write(f"Accuracy/R² Score: {r2}")


        self.plot_results(test.index[seq_length:], y_test_unscaled, predictions_unscaled)

    def plot_results(self, dates, actual, predicted):
        plt.figure(figsize=(10, 6))
        plt.plot(dates, actual, label="Actual River Discharge", color='blue')
        plt.plot(dates, predicted, label="Predicted River Discharge", color='red')
        plt.xlabel("Date")
        plt.ylabel("River Discharge")
        plt.title("Actual vs Predicted River Discharge")
        plt.legend()
        plt.show()
