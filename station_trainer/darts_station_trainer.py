import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import time
import numpy as np
import sklearn
import datetime
import json
import argparse
from typing import Union

# darts imports
from darts import TimeSeries
from darts.models import (
    GaussianProcessFilter, # Filtering model
    KalmanFilter, # Filtering model
    MovingAverageFilter, # Filtering model
    TSMixerModel, # Forecasting model
)
from darts.ad.scorers import (
    NormScorer, 
    KMeansScorer,
    DifferenceScorer,
)
from darts.ad.anomaly_model.filtering_am import FilteringAnomalyModel
from darts.ad.anomaly_model.forecasting_am import ForecastingAnomalyModel
from darts.models.forecasting.forecasting_model import ForecastingModel
from darts.models.filtering.filtering_model import FilteringModel

class DartsTrainer():
    def __init__(self, model_index:int, scorer_index:int, station:str, use_du:bool) -> None:
        self.station = station
        self.use_du = use_du

        # Defining station filepaths
        self.gnss_data_path = f'dataset/{station}/{station}_NEU_train.csv'
        self.gnss_label_path = f'dataset/{station}/{station}_NEU_test_label.csv'
        self.png_path = f'dataset/{station}/{station}_trained_darts.png'
        self.metrics_path = f'dataset/{station}/{station}_metrics_darts.json'

        # Getting the data
        self.gnss_data, self.gnss_label = self.get_data(self.gnss_data_path, self.gnss_label_path)

        model_names = [
            'GaussianProcessFilter',
            'KalmanFilter',
            'MovingAverageFilter',
            'TSMixerModel',
        ]
        model_name = model_names[model_index]

        # Instating a model
        if model_name == 'GaussianProcessFilter':
            model = GaussianProcessFilter()
        elif model_name == 'KalmanFilter':
            model = KalmanFilter()
        elif model_name == 'MovingAverageFilter':
            model = MovingAverageFilter(window=10)
        elif model_name == 'TSMixerModel':
            model = TSMixerModel(input_chunk_length=10, output_chunk_length=1, n_epochs=10)

        # Scorer index (0:Norm, 1:KMeans, 2:Difference)
        self.scorer_index = scorer_index

        # Instantiating scores
        scorers = [
            NormScorer(ord=1),
            KMeansScorer(k=50),
            DifferenceScorer(),
        ]

        if isinstance(model, ForecastingModel):
            self.model = ForecastingAnomalyModel(
                model=model,
                scorer=scorers
            )
            self.forecast = True
        elif isinstance(model, FilteringModel):
            self.model = FilteringAnomalyModel(
                model=model,
                scorer=scorers
            )
            self.forecast = False

    def train(self):
        # Return None in case we don't have data
        if self.gnss_data.empty or  self.gnss_label.empty:
            return None, None, None, None

        self.define_train_label()

        # Training
        start = time.time()
        # Checking if the model is trainable
        allow_model_training = True if hasattr(self.model.model, 'fit') else False
        self.model.fit(self.train, allow_model_training=allow_model_training)
        end = time.time()

        # Elapsed fit time
        fit_time = end - start

        # Decision score
        start = time.time()
        if self.forecast:
            scores = self.model.score(self.train, start=0.0)
        else:
            scores = self.model.score(self.train)
        end = time.time()

        # Elapsed score time
        score_time = end - start

        # Transforming the scores into a numpy array 
        scores = scores[self.scorer_index].data_array().to_numpy()

        # DifferenceScorer return an anomaly scores for each component in axis 1. Calculating the mean
        scores = np.mean(scores, axis=1)
        scores = scores.squeeze()

        # Scaling between 0-1
        scaler = MinMaxScaler(feature_range=(0, 1))
        scores = scaler.fit_transform(scores.reshape(-1, 1)).flatten()

        # Calculating predictions based on a percentile
        percentile = 99.0
        threshold = np.percentile(scores, percentile)
        pred = (scores > threshold).astype('int')

        # Calculationg metrics
        truth = self.label.data_array().to_numpy().squeeze()
        # Adjusting the begining of the truth to be paired with scores
        begining = len(truth) - len(scores)
        truth = truth[begining:]
        precision, recall, f1_score, support = sklearn.metrics.precision_recall_fscore_support(pred, truth)
        accuracy = sklearn.metrics.accuracy_score(pred, truth)
        f1 = sklearn.metrics.f1_score(pred, truth)

        # MSE
        mse = np.mean((scores - truth) ** 2)

        print(f"Accuracy {accuracy}")
        print(f"Precision {precision}")
        print(f"Recall {recall}")
        print(f"F1 score {f1}")
        print(f"MSE {mse}")
        print(f"Elapsed fit time: {fit_time:.2f} seconds")
        print(f"Elapsed score time: {score_time:.2f} seconds")

        metrics = {
            'Type': 'Station',
            'Accuracy':accuracy,
            'Precision':np.array2string(precision, precision=2, separator=', '),
            'Recall':np.array2string(recall, precision=2, separator=', '),
            'F1':f1,
            'MSE':mse,
            'Processing Time:':f'{fit_time+score_time:.2f} seconds'
        }

        return scores, truth, pred, metrics
    
    def define_train_label(self):
        if self.use_du:
            i = 4
        else:
            i = 3
        
        self.train = TimeSeries.from_dataframe(
            df=self.gnss_data, 
            time_col=self.gnss_data.columns[0], 
            value_cols=self.gnss_data.columns[1:i], 
            ).astype(np.float32) # Using float32 because MPS doesn't accepts float62
        self.label = TimeSeries.from_dataframe(
            df=self.gnss_label, 
            time_col=self.gnss_label.columns[0], 
            value_cols=self.gnss_label.columns[1], 
            ).astype(np.float32) # Using float32 because MPS doesn't accepts float62
    
    def get_data(self, gnss_data_path: str, gnss_label_path:str) -> tuple[pd.DataFrame, pd.DataFrame]:
        try:
            gnss_data = pd.read_csv(gnss_data_path)
            gnss_label = pd.read_csv(gnss_label_path)
        except:
            gnss_data, gnss_label = pd.DataFrame(), pd.DataFrame()

        return self.fill_missing_data(gnss_data=gnss_data, gnss_label=gnss_label)

    def fill_missing_data(self, gnss_data, gnss_label):
        # Filling missing data. Creating missing gps weeks and filling with 0
        gnss_data = gnss_data.set_index("gps_week")
        gnss_data = gnss_data.reindex(list(range(gnss_data.index.min(),gnss_data.index.max()+1)),fill_value=0)
        gnss_data = gnss_data.reset_index()

        # Step 1: copy the gps_week and set index
        gnss_label['week_copy'] = gnss_label.gps_week
        gnss_label = gnss_label.set_index("gps_week")

        # Step 2 getting the edge weeks
        edge_week1 = gnss_label.diff(periods = 1)
        edge_week1 = edge_week1[edge_week1.week_copy > 1]
        edge_week2 = gnss_label.diff(periods = -1)
        edge_week2 = edge_week2[edge_week2.week_copy < -1]
        edge_weeks = list(edge_week1.index) + list(edge_week2.index)

        # Filling missing data. Creating missing gps weeks and filling with 0
        gnss_label = gnss_label.reindex(list(range(gnss_label.index.min(),gnss_label.index.max()+1)),fill_value=0)
        gnss_label = gnss_label.reset_index()

        # Setting the edge weeks as  1
        gnss_label.loc[gnss_label['gps_week'].isin(edge_weeks), 'label'] = 1
        gnss_label = gnss_label.drop('week_copy', axis=1)

        return gnss_data, gnss_label

    def plot_experiment(self, scores:np.array, pred:np.array) -> None:
        plt.clf()

        # Create the figure and primary y-axis
        fig, ax1 = plt.subplots()

        # Plot GNSS data on the primary y-axis
        ax1.plot(self.train.time_index, self.train.values()[:, 0], color = 'cornflowerblue', label = 'Series DN')
        ax1.plot(self.train.time_index, self.train.values()[:, 1], color = 'gold', label = 'Series DE')

        # Plotting anomalies
        label = self.label.pd_dataframe()
        anomalies = label[label.label == 1]
        if not anomalies.empty:
            ax1.vlines(anomalies.index, ymin=plt.ylim()[0], ymax=plt.ylim()[1], color = 'black', linestyle='dashed', alpha=0.5, label='Descontinuity')

        label['pred'] = pred
        predictions = label[label.pred == 1]
        if not predictions.empty:
            ax1.vlines(predictions.index, ymin=plt.ylim()[0], ymax=plt.ylim()[1], color = 'red', alpha=0.5, label='Prediction')

        # Create the secondary y-axis (twinx)
        ax2 = ax1.twinx()

        # Plot data3 on the secondary y-axis
        ax2.plot(self.train.time_index, scores, color='black', linewidth=0.5, label='Scores')

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
        plt.close()

    def save_metrics(self, metrics):
        with open(self.metrics_path, 'w') as result:
            json.dump(metrics, result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Process Stations')
    parser.add_argument(
        '-s',
        '-station',
        help='Station name of a single station',
        default='CHEC' # positional argument
    )
    parser.add_argument(
        '-m',
        '-model_index',
        help='Use model_index = 0,1,2,3.',
        choices=[0,1,2,3],
        type=int,
        default=3 # positional argument
    )
    parser.add_argument(
        '-si',
        '-scorer_index',
        help='Scorer index for the Darts filters = 0,1,2.',
        choices=[0,1,2],
        type=int,
        default=0 # positional argument
    )

    parsed_args = parser.parse_args()
    print(f"Running with {parsed_args} parameters.")
    station = parsed_args.s

    trainer = DartsTrainer(
        model_index=parsed_args.m,
        scorer_index=parsed_args.si,
        station=station,
        use_du=False,
    )

    try:
        scores, truth, pred, metrics = trainer.train()

        if (scores is not None) and (truth is not None) and (pred is not None) and (metrics is not None):
            trainer.plot_experiment(scores=scores, pred=pred)

            trainer.save_metrics(metrics=metrics)
    except Exception as e:
        ts = datetime.datetime.now()
        exception_message = str(e.args[0])
        error_message = f'Error processing station: {station}: {exception_message}'
        print(error_message)
        log = {
            'Processing log': {
                'Station':station,
                'Timestamp':ts.strftime('%Y-%m-%d %H:%M:%S'),
                'Error message':error_message
            }
        }
        with open(f'dataset/{station}/{station}_log.json', 'w') as file:
            json.dump(log, file)
