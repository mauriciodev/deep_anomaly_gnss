# stations can be dataset/brazil_stations.txt or dataset/ecuador_stations.txt
# params can be process_stations/params_chec.json or process_stations/params_braz.json (not necessary for darts)
# filtering_model_index can be -1 for TimesNet and 0,1,2 for Darts
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
python -u process_stations/process_stations.py \
  -stations dataset/ecuador_stations.txt \
  -params process_stations/params_chec.json \
  -model_index -1 \