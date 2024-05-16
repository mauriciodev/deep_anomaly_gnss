# deep_anomaly_gnss

## Downloading data from both stations
python preprocess/gnss_preprocessor.py -s dataset/brazil_stations.txt
python preprocess/gnss_preprocessor.py -s dataset/ecuador_stations.txt

## Hyperparameter tunning for BRAZ and CHEC stations
python nni_experiment/nni_config.py -s BRAZ
python nni_experiment/nni_config.py -s CHEC


## Currently we must check the results and copy params manually
<!-- TENHO QUE EDITAR O ARQUIVO process_stations/params.json

nnictl view lwv0o59f
#nnictl experiment export lwv0o59f -f teste.json -t json

#from nni import Experiment

#exp = Experiment.view('lwv0o59f',non_blocking=True)
#data = exp.export_data()-->

# Processing every station in each country with the params
python process_stations/process_stations.py -s dataset/brazil_stations.txt -p process_stations/params_chec.json
python process_stations/process_stations.py -s dataset/ecuador_stations.txt -p process_stations/params_chec.json

python process_stations/process_stations.py -s dataset/brazil_stations.txt -p process_stations/params_braz.json
python process_stations/process_stations.py -s dataset/ecuador_stations.txt -p process_stations/params_braz.json

# DARTS models

python process_stations/process_stations.py -s dataset/brazil_stations.txt -mi 2 -si 1
python process_stations/process_stations.py -s dataset/ecuador_stations.txt -mi 2 -mi 1
