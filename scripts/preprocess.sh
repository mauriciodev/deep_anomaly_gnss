# Downloading data from both stations
# stations (s) can be dataset/brazil_stations.txt or dataset/ecuador_stations.txt
python -u preprocess/gnss_preprocessor.py -s dataset/brazil_stations.txt
python -u preprocess/gnss_preprocessor.py -s dataset/ecuador_stations.txt
