# stations (s) can be dataset/brazil_stations.txt or dataset/ecuador_stations.txt
# model_index (mi):
    # 0 - GaussianProcessFilter
    # 1 - KalmanFilter
    # 2 - MovingAverageFilter
    # 3 - TSMixer
    # 4 - Transformer
# scorer_index (si):
    # 0 - NormScorer
    # 1 - KMeansScorer
    # 2 - DifferenceScorer
# For Ecuador
python -u process_stations/process_stations.py -stations dataset/ecuador_stations.txt -model_index 0 -scorer_index 0
python -u process_stations/process_stations.py -stations dataset/ecuador_stations.txt -model_index 0 -scorer_index 1
python -u process_stations/process_stations.py -stations dataset/ecuador_stations.txt -model_index 0 -scorer_index 2
python -u process_stations/process_stations.py -stations dataset/ecuador_stations.txt -model_index 1 -scorer_index 0
python -u process_stations/process_stations.py -stations dataset/ecuador_stations.txt -model_index 1 -scorer_index 1
python -u process_stations/process_stations.py -stations dataset/ecuador_stations.txt -model_index 1 -scorer_index 2
python -u process_stations/process_stations.py -stations dataset/ecuador_stations.txt -model_index 2 -scorer_index 0
python -u process_stations/process_stations.py -stations dataset/ecuador_stations.txt -model_index 2 -scorer_index 1
python -u process_stations/process_stations.py -stations dataset/ecuador_stations.txt -model_index 2 -scorer_index 2
python -u process_stations/process_stations.py -stations dataset/ecuador_stations.txt -model_index 3 -scorer_index 0
python -u process_stations/process_stations.py -stations dataset/ecuador_stations.txt -model_index 3 -scorer_index 1
python -u process_stations/process_stations.py -stations dataset/ecuador_stations.txt -model_index 3 -scorer_index 2
python -u process_stations/process_stations.py -stations dataset/ecuador_stations.txt -model_index 4 -scorer_index 0
python -u process_stations/process_stations.py -stations dataset/ecuador_stations.txt -model_index 4 -scorer_index 1
python -u process_stations/process_stations.py -stations dataset/ecuador_stations.txt -model_index 4 -scorer_index 2
# For Brazil
python -u process_stations/process_stations.py -stations dataset/brazil_stations.txt -model_index 0 -scorer_index 0
python -u process_stations/process_stations.py -stations dataset/brazil_stations.txt -model_index 0 -scorer_index 1
python -u process_stations/process_stations.py -stations dataset/brazil_stations.txt -model_index 0 -scorer_index 2
python -u process_stations/process_stations.py -stations dataset/brazil_stations.txt -model_index 1 -scorer_index 0
python -u process_stations/process_stations.py -stations dataset/brazil_stations.txt -model_index 1 -scorer_index 1
python -u process_stations/process_stations.py -stations dataset/brazil_stations.txt -model_index 1 -scorer_index 2
python -u process_stations/process_stations.py -stations dataset/brazil_stations.txt -model_index 2 -scorer_index 0
python -u process_stations/process_stations.py -stations dataset/brazil_stations.txt -model_index 2 -scorer_index 1
python -u process_stations/process_stations.py -stations dataset/brazil_stations.txt -model_index 2 -scorer_index 2
python -u process_stations/process_stations.py -stations dataset/brazil_stations.txt -model_index 3 -scorer_index 0
python -u process_stations/process_stations.py -stations dataset/brazil_stations.txt -model_index 3 -scorer_index 1
python -u process_stations/process_stations.py -stations dataset/brazil_stations.txt -model_index 3 -scorer_index 2
python -u process_stations/process_stations.py -stations dataset/brazil_stations.txt -model_index 4 -scorer_index 0
python -u process_stations/process_stations.py -stations dataset/brazil_stations.txt -model_index 4 -scorer_index 1
python -u process_stations/process_stations.py -stations dataset/brazil_stations.txt -model_index 4 -scorer_index 2