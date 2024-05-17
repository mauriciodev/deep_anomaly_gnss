# station or s is a 4 digit station name present in dataset/
# params or p can be process_stations/params_chec.json or process_stations/params_braz.json or any other you want to use
# model_index or mi must be -1
# For Ecuador
python -u process_stations/process_stations.py -s dataset/ecuador_stations.txt -p process_stations/params_braz.json -mi -1
python -u process_stations/process_stations.py -s dataset/ecuador_stations.txt -p process_stations/params_chec.json -mi -1
# For Brazil
python -u process_stations/process_stations.py -s dataset/brazil_stations.txt -p process_stations/params_braz.json -mi -1
python -u process_stations/process_stations.py -s dataset/brazil_stations.txt -p process_stations/params_chec.json -mi -1