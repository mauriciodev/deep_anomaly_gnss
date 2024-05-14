#!/bin/bash
python nni_experiment/nni_config.py -s BRAZ
python nni_experiment/nni_config.py -s CHEC

#TENHO QUE EDITAR O ARQUIVO process_stations/params.json

#nnictl view lwv0o59f
#nnictl experiment export lwv0o59f -f teste.json -t json

#from nni import Experiment

#exp = Experiment.view('lwv0o59f',non_blocking=True)
#data = exp.export_data()

python process_stations/process_stations.py -s dataset/brazil_stations.txt -p process_stations/params_chec.json
python process_stations/process_stations.py -s dataset/ecuador_stations.txt -p process_stations/params_chec.json

python process_stations/process_stations.py -s dataset/brazil_stations.txt -p process_stations/params_braz.json
python process_stations/process_stations.py -s dataset/ecuador_stations.txt -p process_stations/params_braz.json


python process_stations/process_stations.py -s dataset/brazil_stations.txt -f 0
python process_stations/process_stations.py -s dataset/ecuador_stations.txt -f 0

python process_stations/process_stations.py -s dataset/brazil_stations.txt -f 1
python process_stations/process_stations.py -s dataset/ecuador_stations.txt -f 1

python process_stations/process_stations.py -s dataset/brazil_stations.txt -f 2
python process_stations/process_stations.py -s dataset/ecuador_stations.txt -f 2
