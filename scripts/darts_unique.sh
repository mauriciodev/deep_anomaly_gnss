# station (s) is a 4 digit station name present in dataset/
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
python -u station_trainer/darts_station_trainer.py -s BRAZ -mi 1 -si 0