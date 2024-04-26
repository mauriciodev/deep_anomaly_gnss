from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
#from deepod.models.time_series import TimesNet as Model
import sys
sys.path.append('.')
from model.timesnet import TimesNet as Model
import sklearn
import pandas as pd
import numpy as np
import os

class StationTrainer():
    def __init__(self, station:str, use_du:bool) -> None:
        self.station = station
        self.use_du = use_du

        # Defining station data filepaths
        self.gnss_data_path = f'dataset/{station}/{station}_NEU_train.csv'
        self.gnss_label_path = f'dataset/{station}/{station}_NEU_test_label.csv'
        self.png_path = f'dataset/{station}/{station}_trained.png'

        self.gnss_data, self.gnss_label = self.get_data(self.gnss_data_path, self.gnss_label_path)
        
    def get_data(self, gnss_data_path: str, gnss_label_path:str) -> tuple[pd.DataFrame, pd.DataFrame]:
        try:
            gnss_data = pd.read_csv(gnss_data_path)
            gnss_label = pd.read_csv(gnss_label_path)
        except:
            gnss_data, gnss_label = pd.DataFrame(), pd.DataFrame()

        return gnss_data, gnss_label
    
    def get_params(self):
        # Best Hyperparameters
        params = {
            'seq_len': 10, 
            'stride': 1, 
            'lr': 0.0001, 
            'epochs': 10, 
            'batch_size': 128, 
            'epoch_steps': -1, 
            'prt_steps': 1, 
            'device': 'mps', 
            'pred_len': 0, 
            'e_layers': 3, 
            'd_model': 64, 
            'd_ff': 64, 
            'dropout': 0.2, 
            'top_k': 3, 
            'num_kernels': 6, 
            'verbose': 2, 
            'random_state': 42, 
            'percentile': 99, 
            'patience': 3, 
            'delta': 1e-7
        }
        return params
    
    def train(self) -> tuple[np.array, np.array, np.array]:
        # Return None in case we don't have data
        if self.gnss_data.empty or  self.gnss_label.empty:
            return None, None, None
        
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
        model.fit(training_data)
        
        # Getting scores
        scores = model.decision_function(training_data)

        # Calculating predictions based on a percentile
        threshold = np.percentile(scores, params['percentile'])
        pred = (scores > threshold).astype('int').ravel()

        # Calculationg metrics
        truth = self.gnss_label.label.to_numpy().flatten()
        precision, recall, f1_score, support = sklearn.metrics.precision_recall_fscore_support(pred, truth)
        accuracy = sklearn.metrics.accuracy_score(pred, truth)
        f1 = sklearn.metrics.f1_score(pred, truth)
        
        print(f"Accuracy {accuracy}")
        print(f"Precision {precision}")
        print(f"Recall {recall}")
        print(f"F1 score {f1}")

        return scores, truth, pred

    def plot_experiment(self, pred:np.array) -> None:
        plt.clf()
        #plotting data
        plt.plot(self.gnss_data.gps_week, self.gnss_data['dn(m)'], color = 'cornflowerblue', label = 'Series DN')
        plt.plot(self.gnss_data.gps_week, self.gnss_data['de(m)'], color = 'gold', label = 'Series DE')
        plt.plot(self.gnss_data.gps_week, self.gnss_data['du(m)'], color = 'magenta', label = 'Series DU')
        
        self.gnss_label['pred'] = pred
        # Plotting anomalies
        anomalies = self.gnss_label[self.gnss_label.label == 1]
        plt.vlines(anomalies.gps_week, ymin=plt.ylim()[0], ymax=plt.ylim()[1], color = 'black', alpha=0.5, label='Descontinuity')

        # Plotting predictions
        predictions = self.gnss_label[self.gnss_label.pred == 1]
        plt.vlines(predictions.gps_week, ymin=plt.ylim()[0], ymax=plt.ylim()[1], color = 'red', alpha=0.5, label='Prediction')

        plt.legend()
        plt.title(f'Station: {self.station}', loc='center')
        plt.savefig(self.png_path, format='png')

if __name__ == '__main__':
    station = 'BRAZ'
    station_trainer = StationTrainer(station=station, use_du=False)

    scores, truth, pred = station_trainer.train()
    print(scores.shape, truth.shape, pred.shape)

    station_trainer.plot_experiment(pred)