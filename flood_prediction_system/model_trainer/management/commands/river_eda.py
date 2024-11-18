from django.core.management.base import BaseCommand
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from data_collection.models import RiverData

class Command(BaseCommand):
    help = 'Preprocess and visualize river discharge data for analysis'

    def handle(self, *args, **kwargs):
        # Load data from the database
        queryset = RiverData.objects.all().values(
            'date', 'river_discharge', 'river_discharge_mean', 'river_discharge_median', 
            'river_discharge_max', 'river_discharge_min', 'river_discharge_p25', 'river_discharge_p75'
        )

        # Convert queryset to DataFrame
        data = pd.DataFrame(list(queryset))

        if data.empty:
            print("No data found in the database.")
            return

        # Preprocessing
        data['date'] = pd.to_datetime(data['date'])
        data.set_index('date', inplace=True)

        # 1. Summary statistics
        print("\nSummary statistics:")
        print(data.describe())

        # 2. Time series plot for river discharge variables
        plt.figure(figsize=(14, 7))
        plt.plot(data.index, data['river_discharge'], label='River Discharge')
        plt.plot(data.index, data['river_discharge_mean'], label='Mean River Discharge')
        plt.plot(data.index, data['river_discharge_median'], label='Median River Discharge')
        plt.fill_between(data.index, data['river_discharge_min'], data['river_discharge_max'], color='lightgray', alpha=0.5, label='Min-Max Range')
        plt.title('River Discharge Over Time')
        plt.xlabel('Date')
        plt.ylabel('Discharge (m³/s)')
        plt.legend()
        plt.grid(True)
        plt.show()

        # 3. Boxplot to show distribution of river discharge
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='variable', y='value', data=pd.melt(data[['river_discharge', 'river_discharge_mean', 'river_discharge_median']]))
        plt.title('Boxplot of River Discharge Statistics')
        plt.show()

        # 4. Heatmap of correlation between river discharge statistics
        plt.figure(figsize=(10, 8))
        corr_matrix = data[['river_discharge', 'river_discharge_mean', 'river_discharge_median', 'river_discharge_max', 'river_discharge_min']].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Heatmap for River Discharge Variables')
        plt.show()

        # 5. Time series decomposition for river discharge
        decomposition = seasonal_decompose(data['river_discharge'].dropna(), model='additive', period=30)
        decomposition.plot()
        plt.show()

        print("River Data Exploratory Data Analysis completed.")
