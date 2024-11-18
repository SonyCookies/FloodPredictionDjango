from django.shortcuts import render

# your_app/views.py

from django.shortcuts import render
from data_collection.models import WeatherModel  # Import your model
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from matplotlib import pyplot as plt

def run_arima_view(request):
    # Step 1: Query data from WeatherModel
    weather_data = WeatherModel.objects.all().values('datetime', 'temp')
    df = pd.DataFrame.from_records(weather_data)

    # Step 2: Preprocess data (convert 'datetime' and set as index)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)

    # Step 3: Check for stationarity using ADF test
    result = adfuller(df['temp'])
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')

    # Step 4: Fit ARIMA model
    df['diff'] = df['temp'].diff().dropna()

    # Plot ACF and PACF (optional for visualization)
    plot_acf(df['diff'].dropna())
    plot_pacf(df['diff'].dropna())
    plt.show()

    # Step 5: Fit ARIMA model
    model = ARIMA(df['temp'], order=(5, 1, 2))
    model_fit = model.fit()
    print(model_fit.summary())

    # Step 6: Forecast future values
    forecast = model_fit.forecast(steps=10)
    print(forecast)

    # Step 7: Pass forecast to the template (if needed)
    context = {
        'forecast': forecast,
    }
    return render(request, 'arima_model/arima.html', context)
