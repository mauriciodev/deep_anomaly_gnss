import numpy as np
import sklearn
import time
import json
import tqdm
import datetime

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
    for station in (pbar := tqdm.tqdm(stations)):
        try:
            # Set progress bar message
            pbar.set_postfix({'Processing Station': station})

            # Our station trainer
            station_trainer = StationTrainer(station=station, use_du=False)

            scores, truth, pred, metrics = station_trainer.train()
        except Exception as e:
            ts = datetime.datetime.now()
            exception_message = str(e.args[0])
            error_message = f'Error processing station: {station}: {exception_message}'
            print(error_message)
            pbar.write(error_message)
            log = {
                'Processing log': {
                    'Station':{station},
                    'Timestamp':{ts},
                    'Error message':{error_message}
                }
            }
            with open(f'dataset/{station}/{station}_log.txt', 'w') as file:
                json.dump(log, file)
                
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

    # Calculation global metrics
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

    # Saving the global metrics file
    with open('dataset/global_metrics.txt', 'w') as result:
        json.dump(metrics, result)

if __name__ == '__main__':
    stations_filepath = 'dataset/brazil_stations.txt'
    stations = read_stations_file(stations_filepath)
    # Sample stations to check the code
    #stations = ['BRAZ', 'CHEC']

    process_stations(stations=stations)
