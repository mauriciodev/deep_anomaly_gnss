import numpy as np
import pandas as pd
import sklearn
import nni
import pandas as pd
import numpy as np
import logging
import os
from sklearn.preprocessing import MinMaxScaler
import argparse
import torch

# Importing our custom TimesNet with Convergence Early Stop
import sys
sys.path.append('.')
from model.timesnet import TimesNet as Model

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
    scaler = MinMaxScaler(feature_range=(0, 1))
    scores = scaler.fit_transform(scores.reshape(-1, 1)).flatten()     

    # Calculating predictions based on a percentile
    threshold = np.percentile(scores, params['percentile'])
    pred = (scores > threshold).astype('int').ravel()

    # Calculationg metrics
    truth = gnss_label.label.to_numpy().flatten()
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

    return mse

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
        'pred_len': 0, 
        'e_layers': 3, 
        'd_model': 128, 
        'd_ff': 128, 
        'dropout': 0.4, 
        'top_k': 5, 
        'num_kernels': 6, 
        'verbose': 2, 
        'random_state': 42, 
        'percentile': 99.0, 
        'patience': 3, 
        'delta': 1e-7
    }
    if torch.backends.mps.is_available(): params['device']='mps'
    params.update(nni.get_next_parameter())
    
    # Logging
    logging.info(params)

    # Training
    f1 = train(
        station=station,
        params=params, 
        gnss_data_path=gnss_data_path, 
        gnss_label_path=gnss_label_path,
        use_du=use_du,
    )

    # Reporting f1 to nni so it can be tracked
    nni.report_final_result(f1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='experiment')
    parser.add_argument(
        '-s',
        '-station',
        help='Station name',
        default='BRAZ' # positional argument
    )
    station = parser.parse_args().s
    exec_process(station, False)