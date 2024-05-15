# stations can be dataset/brazil_stations.txt or dataset/ecuador_stations.txt
# params can be process_stations/params_chec.json or process_stations/params_braz.json (not necessary for darts)
# model_index can be -1 for TimesNet and 0,1,2 for Darts
    # -1 - TimesNet (Choosing this will ignore scorer_index)
    # 0 - GaussianProcessFilter
    # 1 - KalmanFilter
    # 2 - MovingAverageFilter
    # 3 - TSMixer
    # 4 - Transformer
# scorer_index can be 0,1,2 (not necessary for TimesNet)
    # 0 - NormScorer
    # 1 - KMeansScorer
    # 2 - DifferenceScorer
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