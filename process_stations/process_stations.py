import numpy as np
import sklearn
import time

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
    total_truth = []
    total_pred = []
    start = time.time()
    for station in stations:
        # Our station trainer
        station_trainer = StationTrainer(station=station, use_du=False)

        # Actual training
        try:
            scores, truth, pred, metrics = station_trainer.train()
        except:
            continue

        # Gathering station truth and pred for stacking
        total_truth.append(truth)
        total_pred.append(pred)

        # Ploting the graph in the dataset/{station} folder
        station_trainer.plot_experiment(scores=scores, pred=pred)

        # Saving Station metrics
        station_trainer.save_metrics(metrics=metrics)
    end = time.time()
    
    # Elapsed time
    elapsed_time = end - start

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
    print(f"Processing time: {elapsed_time:.2f} seconds")

    metrics = {
        'Type': 'Global',
        'Accuracy':accuracy,
        'Precision':np.array2string(precision, precision=2, separator=','),
        'Recall':np.array2string(recall, precision=2, separator=','),
        'F1':f1,
        'Processing Time:':f'{elapsed_time:.2f} seconds'
    }

if __name__ == '__main__':
    stations_filepath = 'dataset/brazil_stations.txt'
    stations = read_stations_file(stations_filepath)
    #stations = ['BRAZ', 'CHEC']

    process_stations(stations=stations)
