from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from deepod.models.time_series import TimesNet as Model
import sklearn
import nni
import pandas as pd
import numpy as np
import logging

def get_data(gnss_data_path: str, gnss_label_path:str) -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv(gnss_data_path), pd.read_csv(gnss_label_path)

def train(params: dict, gnss_data_path: str, gnss_label_path:str) -> float:
    model_params = {key: value for key, value in params.items() if key != 'percentile'}

    gnss_data, gnss_label = get_data(gnss_data_path, gnss_label_path)
    
    # Instantiationg and fitting model
    model = Model(**model_params)
    model.fit(gnss_data.iloc[:, [1,2]])
    
    # Getting scores
    scores = model.decision_function(gnss_data.iloc[:, [1,2]])

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

    return f1

if __name__ == '__main__':
    
    gnss_data_path = 'dataset/NEU/train.csv'
    gnss_label_path = 'dataset/NEU/test_label.csv'
    
    # Hyperparameters
    params = {
        'seq_len':10,
        'stride':1,
        'lr':1e-4,
        'epochs':30,
        'batch_size':64,
        'epoch_steps':20,
        'prt_steps':1,
        'pred_len':0,
        'e_layers':2,
        'd_model':64,
        'd_ff':64,
        'dropout':0.1,
        'top_k':5,
        'num_kernels':6,
        'verbose':2,
        'random_state':42,
        'percentile':98,
        'device':'mps',
    }
    params.update(nni.get_next_parameter())
    
    # Logging
    logging.info(params)
    
    # Training
    f1 = train(
        params=params, 
        gnss_data_path=gnss_data_path, 
        gnss_label_path=gnss_label_path, 
    )
    
    # Reporting f1 to nni so it can be tracked
    nni.report_final_result(f1)
