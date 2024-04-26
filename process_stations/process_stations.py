import numpy as np
import sklearn
import json

import sys
sys.path.append('.')
from station_trainer.station_trainer import StationTrainer

def read_stations_file(stations_file:str) -> list:
    try:
        with open(stations_file) as file:
            content = file.read()
            content = content.replace(' ', '')
            return content.split(',')
    except FileNotFoundError as e:
        print(f"Error: File '{stations_file}' not found.")
        return []
    except Exception as e:
        print(f"Error reading file '{stations_file}': {e}")
        return []
        
def process_stations(stations:list):
    #stations = ['BRAZ', 'CHEC']
    total_truth = []
    total_pred = []
    for station in stations:
        # Our station trainer
        station_trainer = StationTrainer(station=station, use_du=False)

        # Actual training
        try:
            scores, truth, pred = station_trainer.train()
        except:
            continue
        total_truth.append(truth)
        total_pred.append(pred)

        # Ploting the graph in the dataset/{station} folder
        station_trainer.plot_experiment(scores=scores, pred=pred)

    # Stacking truth and pred
    stacked_truth = np.hstack([*total_truth])
    stacked_pred = np.hstack([*total_pred])

    precision, recall, f1_score, support = sklearn.metrics.precision_recall_fscore_support(stacked_pred, stacked_truth)
    accuracy = sklearn.metrics.accuracy_score(stacked_pred, stacked_truth)
    f1 = sklearn.metrics.f1_score(stacked_pred, stacked_truth)
    
    print(f"Global Accuracy: {accuracy}")
    print(f"Global Precision: {precision}")
    print(f"Global Recall: {recall}")
    print(f"Global F1 score: {f1}")

    metrics = {
        'Type': 'Global',
        'Accuracy':accuracy,
        'Precision':np.array2string(precision, precision=2, separator=', '),
        'Recall':np.array2string(recall, precision=2, separator=', '),
        'F1':f1,
    }

    with open('dataset/global_metrics.txt', 'w') as result:
        json.dump(metrics, result)

if __name__ == '__main__':
    stations_filepath = 'dataset/brazil_stations.txt'
    stations = read_stations_file(stations_filepath)

    process_stations(stations=stations)