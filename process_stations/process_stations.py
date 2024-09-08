import numpy as np
import pandas as pd
import sklearn
import time
import json
import tqdm
import datetime
import argparse
from pathlib import Path
import sys
sys.path.append('.')
from station_trainer.timesnet_station_trainer import StationTrainer
from station_trainer.darts_station_trainer import DartsTrainer

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
    
def process_stations(stations:list, output_file:str, model_index:int=-1, scorer_index:int=0, params={}):
    total_truth = []
    total_pred = []
    total_scores = []
    total_kpred = []
    start = time.time()
    for station in (pbar := tqdm.tqdm(stations)):
        try:
            # Set progress bar message
            pbar.set_postfix({'Processing Station': station})

            # Our station trainer
            if model_index == -1:
                station_trainer = StationTrainer(
                    station=station, 
                    use_du=False,
                    arg_params=params
                )
            else:
                station_trainer = DartsTrainer(
                    model_index=model_index, 
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
                
                K = np.unique(truth, return_counts=True)[1][1] #number of 1s in the truth
                topK = np.argpartition(scores,-K)[-K:]
                kpred = np.zeros_like(pred)
                kpred[topK]=1
                total_kpred.append(kpred)

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
    stacked_kpred = np.hstack([*total_kpred])

    # Calculation global metrics
    precision, recall, f1_score, support = sklearn.metrics.precision_recall_fscore_support(stacked_truth, stacked_pred)
    accuracy = sklearn.metrics.accuracy_score(stacked_truth, stacked_pred)

    # Calculating global MSE
    mse = np.mean((stacked_scores - stacked_truth) ** 2)
        
    # Calculating top K metrics
    topK_precision, topK_recall, topK_f1_score, topK_support = sklearn.metrics.precision_recall_fscore_support(stacked_truth, stacked_kpred)
    print(sklearn.metrics.classification_report(stacked_truth, stacked_kpred))
    print(sklearn.metrics.confusion_matrix(stacked_truth, stacked_kpred))

    print(f"Global Accuracy: {accuracy}")
    print(f"Global Precision: {precision[1]}")
    print(f"Global Recall: {recall[1]}")
    print(f"Global F1 score: {f1_score[1]}")
    print(f"Global TopK_F1 score: {topK_f1_score[1]}")
    print(f"Global MSE score: {mse}")
    print(f"Processing time: {elapsed_time:.2f} seconds")
    
    if model_index != -1:
        experiment_name = f"{output_file} filter {model_index} scorer {scorer_index}"
    else:
        experiment_name = f"{output_file} {params}"
    metrics = {
        'Experiment': [experiment_name],
        'Scorer': [scorer_index],
        'Filter model': [model_index],
        'Type': ['Global'],
        'Accuracy':[accuracy],
        'Precision':[precision[1]],
        'Recall': [recall[1]],
        'F1':[f1_score[1]],
        'TopK_F1':[topK_f1_score[1]],
        'MSE':[mse],
        'Processing Time:':[f'{elapsed_time:.2f}'],
    }

    # Saving the global metrics file
    new_df = pd.DataFrame.from_dict(metrics)
    if Path(output_file).is_file():
        df = pd.read_csv(output_file)
        df = df._append(new_df)
    else:
        df = new_df
    df.to_csv(output_file, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Process Stations')
    parser.add_argument(
        '-s',
        '-stations',
        help='Station.txt file. A list of 4 digit SIRGAS station codes, separated by comma.',
        default='dataset/ecuador_stations.txt' # positional argument
    )
    parser.add_argument(
        '-p',
        '-params',
        help='params.json file. To control the model\'s hyperparameters.',
        default='' # positional argument
    )
    parser.add_argument(
        '-mi',
        '-model_index',
        help='Use model_index = -1 for TimesNet/ Use model_index = 0,1,2,3,4,5 for Darts.',
        choices=[-1,0,1,2,3,4,5],
        type=int,
        default=-1 # positional argument
    )
    parser.add_argument(
        '-si',
        '-scorer_index',
        help='Scorer index for the Darts filters = 0,1,2 for Darts. Not used on TimesNet.',
        choices=[0,1,2],
        type=int,
        default=2 # positional argument
    )

    parsed_args = parser.parse_args()
    print(f"Running with {parsed_args} parameters.")
    
    stations_filepath = parsed_args.s
    stations = read_stations_file(stations_filepath)
    # Sample stations to check the code
    #stations = ['BRAZ', 'CHEC']
    file_name = (Path(stations_filepath).stem)+'_global_metrics.csv'

    paramsFile = parsed_args.p
    if paramsFile == '':
        params = {}
    else:
        with open(paramsFile, 'r') as f:
            params = json.load(f)


    # Use filtering_model_index = -1 for TimesNet/ Use filtering_model_index = 0,1,2 for Darts
    process_stations(stations=stations, output_file=f'dataset/{file_name}', model_index=parsed_args.mi, scorer_index=parsed_args.si, params=params)
