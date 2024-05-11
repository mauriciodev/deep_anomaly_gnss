import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import time
import numpy as np

# darts imports
from darts import TimeSeries
from darts.metrics import mse as MSE
from darts.models import (
    GaussianProcessFilter,
    KalmanFilter,
    MovingAverageFilter,
)
from darts.ad.scorers import (
    NormScorer, 
    KMeansScorer,
    DifferenceScorer,
)
from darts.ad.anomaly_model.filtering_am import FilteringAnomalyModel

class DartsTrainer():
    def __init__(self, model, scorers, station:str, use_du:bool) -> None:
        self.station = station
        self.use_du = use_du

        # Defining station filepaths
        self.gnss_data_path = f'dataset/{station}/{station}_NEU_train.csv'
        self.gnss_label_path = f'dataset/{station}/{station}_NEU_test_label.csv'
        self.png_path = f'dataset/{station}/{station}_trained_darts.png'
        self.metrics_path = f'dataset/{station}/{station}_metrics_darts.json'

        # Getting the data
        self.gnss_data, self.gnss_label = self.get_data(self.gnss_data_path, self.gnss_label_path)

        # Instantiate the anomaly model with: one forecasting model, and one or more scorers (and corresponding parameters)
        self.anomaly_model = FilteringAnomalyModel(
            model=model,
            scorer=scorers,
        )

    def train(self):
        self.define_train_label()

        # Training
        start = time.time()
        self.anomaly_model.fit(self.train)
        end = time.time()

        # Elapsed fit time
        fit_time = end - start

        # Decision score
        start = time.time()
        scores, pred = self.anomaly_model.score(self.train, return_model_prediction=True)
        end = time.time()

        # Elapsed score time
        score_time = end - start

        scores = scores[0].data_array().to_numpy().squeeze()
        scaler = MinMaxScaler(feature_range=(0, 1))
        scores = scaler.fit_transform(scores.reshape(-1, 1)).flatten()

        # MSE
        mse = MSE(self.train, pred)

        print(f"Elapsed time to fit: {fit_time:.2f} seconds")
        print(f"Elapsed time to score: {score_time:.2f} seconds")
        print(f'MSE: {mse}')

        return scores, pred
    
    def define_train_label(self):
        if self.use_du:
            i = 4
        else:
            i = 3
        
        self.train = TimeSeries.from_dataframe(
            df=self.gnss_data, 
            time_col=self.gnss_data.columns[0], 
            value_cols=self.gnss_data.columns[1:i], 
            fill_missing_dates=True, 
            freq=1, 
            fillna_value=0.0,
            )
        self.label = TimeSeries.from_dataframe(
            df=self.gnss_label, 
            time_col=self.gnss_label.columns[0], 
            value_cols=self.gnss_label.columns[1], 
            fill_missing_dates=True, 
            freq=1, 
            fillna_value=1.0,
            )
    
    def get_data(self, gnss_data_path: str, gnss_label_path:str) -> tuple[pd.DataFrame, pd.DataFrame]:
        try:
            gnss_data = pd.read_csv(gnss_data_path)
            gnss_label = pd.read_csv(gnss_label_path)
        except:
            gnss_data, gnss_label = pd.DataFrame(), pd.DataFrame()

        return gnss_data, gnss_label    

    def plot_experiment(self, scores:np.array, pred) -> None:
        plt.clf()

        # Create the figure and primary y-axis
        fig, ax1 = plt.subplots()

        # Plot GNSS data on the primary y-axis
        ax1.plot(self.train.time_index, self.train.values()[:, 0], color = 'cornflowerblue', label = 'Series DN')
        ax1.plot(self.train.time_index, self.train.values()[:, 1], color = 'gold', label = 'Series DE')

        # Plot predictions
        ax1.plot(self.train.time_index, pred.values()[:, 0], color = 'cornflowerblue', linestyle='dotted', label = 'Pred DN')
        ax1.plot(self.train.time_index, pred.values()[:, 1], color = 'gold', linestyle='dotted', label = 'Pred DE')

        # Plotting anomalies
        anomalies = self.gnss_label[self.gnss_label.label == 1]
        if not anomalies.empty:
            ax1.vlines(anomalies.gps_week, ymin=plt.ylim()[0], ymax=plt.ylim()[1], color = 'black', linestyle='dashed', alpha=0.5, label='Descontinuity')

        # Create the secondary y-axis (twinx)
        ax2 = ax1.twinx()

        # Plot data3 on the secondary y-axis
        ax2.plot(self.train.time_index, scores, color='red', linewidth=0.5, label='Scores')

        # Set labels for axes
        ax1.set_xlabel('GPS Week')
        ax1.set_ylabel('Deviation in meters (m)')
        ax2.set_ylabel('Normalized Score [0-1]')

        # Add legend using all lines
        # ask matplotlib for the plotted objects and their labels
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc=0)

        # Save the plot
        plt.title(f'Station: {self.station}', loc='center')
        plt.savefig(self.png_path, format='png')

if __name__ == '__main__':
    station = 'BRAZ'

    filtering_model_names = ['GaussianProcessFilter', 'KalmanFilter','MovingAverageFilter']
    filtering_model_name = filtering_model_names[2]

    # Instatiate of a filtering model
    if filtering_model_name == 'GaussianProcessFilter':
        filtering_model = GaussianProcessFilter()
    elif filtering_model_name == 'KalmanFilter':
        filtering_model = KalmanFilter()
    else:
        filtering_model = MovingAverageFilter(window=10)

    scorers = [
        NormScorer(ord=1),
        KMeansScorer(k=50),
        DifferenceScorer(),
    ]
    
    trainer = DartsTrainer(
        model=filtering_model,
        scorers=scorers,
        station=station,
        use_du=False,
    )

    scores, pred = trainer.train()

    trainer.plot_experiment(scores, pred)