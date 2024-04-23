from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from deepod.models.time_series import TimesNet as Model
import sklearn
import pandas as pd
import numpy as np

def plot_experiment(gnss_data, gnss_label, pred) -> None:
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
    plt.show()

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

    plot_experiment(gnss_data, gnss_label, pred)

    return f1

if __name__ == '__main__':
    
    gnss_data_path = 'dataset/NEU/train.csv'
    gnss_label_path = 'dataset/NEU/test_label.csv'
    
    # Best Hyperparameters
    params = {
        'seq_len':20,
        'stride':1,
        'lr':1.3e-4,
        'epochs':30,
        'batch_size':64,
        'epoch_steps':40,
        'prt_steps':1,
        'pred_len':0,
        'e_layers':2,
        'd_model':256,
        'd_ff':64,
        'dropout':0.15,
        'top_k':7,
        'num_kernels':6,
        'verbose':2,
        'random_state':42,
        'percentile':99.49872516813187,
        'device':'mps',
        'patience':3,
        'delta':2e-6,
    }
    
    # Training
    f1 = train(
        params=params, 
        gnss_data_path=gnss_data_path, 
        gnss_label_path=gnss_label_path, 
    )