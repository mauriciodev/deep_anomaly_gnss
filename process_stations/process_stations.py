import numpy as np
import sklearn
import time
import json
import tqdm
import datetime
import argparse
from pathlib import Path
import sys
sys.path.append('.')
from station_trainer.station_trainer import StationTrainer
from comparisons.anomaly_detection_darts import DartsTrainer

# darts imports
from darts.models import (
    GaussianProcessFilter,
    KalmanFilter,
    MovingAverageFilter,
)
from darts.ad.scorers import (
    NormScorer, 
    KMeansScorer,
    DifferenceScorer,
)

def read_stations_file(stations_file:str) -> list:
    try:
        with open(stations_file) as file:
            content = file.read()
            content = content.replace(' ', '').strip()
            return content.split(',')
    except FileNotFoundError as e:
        print(f"Error: File '{stations_file}' not found.")
        return []
    except Exception as e:
        print(f"Error reading file '{stations_file}': {e}")
        return []
    
def prepare_darts(filtering_model_index:int):
    filtering_model_names = ['GaussianProcessFilter', 'KalmanFilter','MovingAverageFilter']
    filtering_model_name = filtering_model_names[filtering_model_index]

    # Instatiate of a filtering model
    if filtering_model_name == 'GaussianProcessFilter':
        filtering_model = GaussianProcessFilter()
    elif filtering_model_name == 'KalmanFilter':
        filtering_model = KalmanFilter()
    else:
        filtering_model = MovingAverageFilter(window=10)

    scorers = [
        NormScorer(ord=1),
        KMeansScorer(k=50),
        DifferenceScorer(),
    ]
    return filtering_model, scorers 
        
def process_stations(stations:list, output_file:str, filtering_model_index:int=-1, scorer_index:int=0):
    total_truth = []
    total_pred = []
    total_scores = []
    start = time.time()
    for station in (pbar := tqdm.tqdm(stations)):
        try:
            # Set progress bar message
            pbar.set_postfix({'Processing Station': station})

            # Our station trainer
            if filtering_model_index == -1:
                station_trainer = StationTrainer(
                    station=station, 
                    use_du=False,
                )
            else:
                filtering_model, scorers = prepare_darts(filtering_model_index)
                station_trainer = DartsTrainer(
                    model=filtering_model, 
                    scorers=scorers, 
                    scorer_index=scorer_index, 
                    station=station, 
                    use_du=False,
                )

            scores, truth, pred, metrics = station_trainer.train()
            if (scores is not None) and (truth is not None) and (pred is not None) and (metrics is not None):
                # Gathering station scores, truth and pred for stacking
                total_scores.append(scores)
                total_truth.append(truth)
                total_pred.append(pred)

                # Ploting the graph in the dataset/{station} folder
                station_trainer.plot_experiment(scores=scores, pred=pred)

                # Saving Station metrics
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
            with open(f'dataset/{station}/{station}_log.txt', 'w') as file:
                json.dump(log, file)
                
            continue
    end = time.time()

    # Elapsed time
    elapsed_time = end - start

    # Stacking scores, truth and pred
    stacked_scores = np.hstack([*total_scores])
    stacked_truth = np.hstack([*total_truth])
    stacked_pred = np.hstack([*total_pred])

    # Calculation global metrics
    precision, recall, f1_score, support = sklearn.metrics.precision_recall_fscore_support(stacked_pred, stacked_truth)
    accuracy = sklearn.metrics.accuracy_score(stacked_pred, stacked_truth)
    f1 = sklearn.metrics.f1_score(stacked_pred, stacked_truth)

    # Calculating global MSE
    mse = np.mean((stacked_scores - stacked_truth) ** 2)
    
    print(f"Global Accuracy: {accuracy}")
    print(f"Global Precision: {precision}")
    print(f"Global Recall: {recall}")
    print(f"Global F1 score: {f1}")
    print(f"Global MSE score: {mse}")
    print(f"Processing time: {elapsed_time:.2f} seconds")

    metrics = {
        'Type': 'Global',
        'Accuracy':accuracy,
        'Precision':np.array2string(precision, precision=2, separator=','),
        'Recall':np.array2string(recall, precision=2, separator=','),
        'F1':f1,
        'MSE':mse,
        'Processing Time:':f'{elapsed_time:.2f} seconds'
    }

    # Saving the global metrics file
    with open(output_file, 'w') as result:
        json.dump(metrics, result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Process Stations')
    parser.add_argument(
        '-s',
        '-stations',
        help='Station.txt file. A list of 4 digit SIRGAS station codes, separated by comma.',
        default='dataset/brazil_stations.txt' # positional argument
    )           
    stations_filepath = parser.parse_args().s
    stations = read_stations_file(stations_filepath)
    # Sample stations to check the code
    #stations = ['BRAZ', 'CHEC']
    file_name = (Path(stations_filepath).stem)+'_global_metrics.json'
    process_stations(stations=stations, output_file=f'dataset/{file_name}', filtering_model_index=0, scorer_index=0)
