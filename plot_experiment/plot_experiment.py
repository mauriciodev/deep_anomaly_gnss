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

def plot_experiment(title, gnss_data, gnss_label, pred) -> None:
    plt.clf()
    #plotting data
    plt.plot(gnss_data.gps_week, gnss_data['dn(m)'], color = 'cornflowerblue', label = 'Series DN')
    plt.plot(gnss_data.gps_week, gnss_data['de(m)'], color = 'gold', label = 'Series DE')
    plt.plot(gnss_data.gps_week, gnss_data['du(m)'], color = 'magenta', label = 'Series DU')
    
    gnss_label['pred'] = pred
    # Plotting anomalies
    anomalies = gnss_label[gnss_label.label == 1]
    plt.vlines(anomalies.gps_week, ymin=plt.ylim()[0], ymax=plt.ylim()[1], color = 'black', alpha=0.5, label='Descontinuity')

    # Plotting predictions
    predictions = gnss_label[gnss_label.pred == 1]
    plt.vlines(predictions.gps_week, ymin=plt.ylim()[0], ymax=plt.ylim()[1], color = 'red', alpha=0.5, label='Prediction')

    plt.legend()
    plt.title(f'Station: {title}', loc='center')
    plt.show()

def get_data(gnss_data_path: str, gnss_label_path:str) -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv(gnss_data_path), pd.read_csv(gnss_label_path)

def train(station:str, params: dict, gnss_data_path: str, gnss_label_path:str, use_du:bool) -> float:
    model_params = {key: value for key, value in params.items() if key != 'percentile'}

    gnss_data, gnss_label = get_data(gnss_data_path, gnss_label_path)
    
    # Instantiationg and fitting model
    model = Model(**model_params)

    # Defining training data
    if use_du:
        training_data = gnss_data
    else:
        training_data = gnss_data.iloc[:, [1,2]]

    # Training
    model.fit(training_data)
    
    # Getting scores
    scores = model.decision_function(training_data)

    # Calculating predictions based on a percentile
    threshold = np.percentile(scores, params['percentile'])
    pred = (scores > threshold).astype('int').ravel()

    # Calculationg metrics
    truth = gnss_label.label.to_numpy().flatten()
    precision, recall, f1_score, support = sklearn.metrics.precision_recall_fscore_support(pred, truth)
    accuracy = sklearn.metrics.accuracy_score(pred, truth)
    f1 = sklearn.metrics.f1_score(pred, truth)
    
    print(f"Accuracy {accuracy}")
    print(f"Precision {precision}")
    print(f"Recall {recall}")
    print(f"F1 score {f1}")

    plot_experiment(station, gnss_data, gnss_label, pred)

    return f1

def check_station_data(filepath:str) -> bool:
    if os.path.exists(filepath):
        return True
    else:
        return False
    
def exec_process(station:str, use_du:bool = False):
    # Defining station data filepaths
    gnss_data_path = f'dataset/{station}/{station}_NEU_train.csv'
    gnss_label_path = f'dataset/{station}/{station}_NEU_test_label.csv'

    if not check_station_data(gnss_data_path) or  not check_station_data(gnss_label_path):
        return
    
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
    # Training
    f1 = train(
        station=station,
        params=params, 
        gnss_data_path=gnss_data_path, 
        gnss_label_path=gnss_label_path,
        use_du=False,
    )

if __name__ == '__main__':
    exec_process('BRAZ', False)