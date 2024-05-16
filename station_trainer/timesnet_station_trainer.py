from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.preprocessing import MinMaxScaler
import json
import time
import datetime
import argparse

# Importing our custom TimesNet with Convergence Early Stop
import sys
sys.path.append('.')
from model.timesnet import TimesNet as Model

class StationTrainer():
    def __init__(self, station:str, use_du:bool, arg_params={}) -> None:
        self.station = station
        self.use_du = use_du

        # Defining station filepaths
        self.gnss_data_path = f'dataset/{station}/{station}_NEU_train.csv'
        self.gnss_label_path = f'dataset/{station}/{station}_NEU_test_label.csv'
        self.png_path = f'dataset/{station}/{station}_trained.pdf'
        self.metrics_path = f'dataset/{station}/{station}_metrics.json'

        # GNSS data
        self.gnss_data, self.gnss_label = self.get_data(self.gnss_data_path, self.gnss_label_path)

        # Hyperparameters
        self.params = {
            'seq_len': 10, 
            'stride': 1, 
            'lr': 0.0001, 
            'epochs': 10, 
            'batch_size': 128, 
            'epoch_steps': -1, 
            'prt_steps': 1, 
            'pred_len': 0, 
            'e_layers': 3, 
            'd_model': 128, 
            'd_ff': 128, 
            'dropout': 0.25, 
            'top_k': 5, 
            'num_kernels': 6, 
            'verbose': 2, 
            'random_state': 42, 
            'percentile': 99.0, 
            'patience': 3, 
            'delta': 1e-7
        }
        if torch.backends.mps.is_available(): self.params['device']='mps'
        self.params.update(arg_params)

    def get_quakes(self):
        return pd.read_csv('dataset/quakes.csv')
        
    def get_data(self, gnss_data_path: str, gnss_label_path:str) -> tuple[pd.DataFrame, pd.DataFrame]:
        try:
            gnss_data = pd.read_csv(gnss_data_path)
            gnss_label = pd.read_csv(gnss_label_path)
        except:
            gnss_data, gnss_label = pd.DataFrame(), pd.DataFrame()

        return gnss_data, gnss_label

    def get_params(self):
        # Best Hyperparameters
        return self.params
    
    def train(self) -> tuple[np.array, np.array, np.array]:
        # Return None in case we don't have data
        if self.gnss_data.empty or  self.gnss_label.empty:
            return None, None, None, None
        
        # Getting Training parameters
        params = self.get_params()

        # Defining model parameters
        model_params = {key: value for key, value in params.items() if key != 'percentile'}
        
        # Instantiationg and fitting model
        model = Model(**model_params)

        # Defining training data
        if self.use_du:
            training_data = self.gnss_data
        else:
            training_data = self.gnss_data.iloc[:, [1,2]]

        # Training
        start = time.time()
        model.fit(training_data)
        end = time.time()

        # Elapsed time
        elapsed_time = end - start
        
        # Getting scores
        scores = model.decision_function(training_data)

        # Scaling
        scaler = MinMaxScaler(feature_range=(0, 1))
        scores = scaler.fit_transform(scores.reshape(-1, 1)).flatten()

        # Calculating predictions based on a percentile
        threshold = np.percentile(scores, params['percentile'])
        pred = (scores > threshold).astype('int')

        # Calculationg metrics
        truth = self.gnss_label.label.to_numpy()
        precision, recall, f1_score, support = sklearn.metrics.precision_recall_fscore_support(pred, truth)
        accuracy = sklearn.metrics.accuracy_score(pred, truth)
        f1 = sklearn.metrics.f1_score(pred, truth)

        # Calculating MSE
        mse = np.mean((scores - truth) ** 2)
        
        print(f"Accuracy {accuracy}")
        print(f"Precision {precision}")
        print(f"Recall {recall}")
        print(f"F1 score {f1}")
        print(f"MSE {mse}")
        print(f"Elapsed time: {elapsed_time:.2f} seconds")

        metrics = {
            'Type': 'Station',
            'Accuracy':accuracy,
            'Precision':np.array2string(precision, precision=2, separator=', '),
            'Recall':np.array2string(recall, precision=2, separator=', '),
            'F1':f1,
            'MSE':mse,
            'Processing Time:':f'{elapsed_time:.2f} seconds'
        }

        return scores, truth, pred, metrics

    def plot_experiment(self, scores:np.array, pred:np.array) -> None:
        plt.clf()

        # Create the figure and primary y-axis
        fig, ax1 = plt.subplots()

        # Plot GNSS data on the primary y-axis
        ax1.plot(self.gnss_data.gps_week, self.gnss_data['dn(m)'], color = 'cornflowerblue', label = 'Series DN')
        ax1.plot(self.gnss_data.gps_week, self.gnss_data['de(m)'], color = 'gold', label = 'Series DE')
        ax1.plot(self.gnss_data.gps_week, self.gnss_data['du(m)'], color = 'magenta', label = 'Series DU')

        self.gnss_label['pred'] = pred
        # Plotting anomalies
        anomalies = self.gnss_label[self.gnss_label.label == 1]
        if not anomalies.empty:
            ax1.vlines(anomalies.gps_week, ymin=plt.ylim()[0], ymax=plt.ylim()[1], color = 'black', linestyle='dashed', alpha=0.5, label='Descontinuity')

        # Plotting predictions
        predictions = self.gnss_label[self.gnss_label.pred == 1]
        if not predictions.empty:
            ax1.vlines(predictions.gps_week, ymin=plt.ylim()[0], ymax=plt.ylim()[1], color = 'red', alpha=0.5, label='Prediction')

        # Plotting quakes
        """     
        quakes = self.get_quakes()
        quakes = quakes[quakes.mag >= 3.0]
        quakes = quakes[quakes.gps_week.isin(predictions.gps_week)]
        ax1.vlines(quakes.gps_week, ymin=plt.ylim()[0], ymax=plt.ylim()[1], color = 'purple', alpha=0.5, label='Quakes >= 4.0')
        """
        # Create the secondary y-axis (twinx)
        ax2 = ax1.twinx()

        # Plot data3 on the secondary y-axis
        ax2.plot(self.gnss_data.gps_week, scores, color='black', linewidth=0.5, label='Scores')

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
        plt.savefig(self.png_path, format='pdf')

    def save_metrics(self, metrics):
        with open(self.metrics_path, 'w') as result:
            json.dump(metrics, result)        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Process Stations')
    parser.add_argument(
        '-s',
        '-station',
        help='Station name of a single station',
        default='CEFT' # positional argument
    )
    parser.add_argument(
        '-p',
        '-params',
        help='params.json file. To control the model\'s hyperparameters.',
        default='' # positional argument
    )

    parsed_args = parser.parse_args()
    print(f"Running with {parsed_args} parameters.")

    station = parsed_args.s

    paramsFile = parsed_args.p
    if paramsFile == '':
        params = {}
    else:
        with open(paramsFile, 'r') as f:
            params = json.load(f)

    station_trainer = StationTrainer(station=station, use_du=False, arg_params=params)

    try:
        scores, truth, pred, metrics = station_trainer.train()

        if (scores is not None) and (truth is not None) and (pred is not None) and (metrics is not None):
            station_trainer.plot_experiment(scores=scores, pred=pred)

            station_trainer.save_metrics(metrics=metrics)
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
