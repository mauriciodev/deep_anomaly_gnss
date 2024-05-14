# station 4 digit station name present in dataset/
# filtering_model_index 0,1,2
    # 0 - GaussianProcessFilter
    # 1 - KalmanFilter
    # 2 - MovingAverageFilter
# scorer_index can be 0,1,2
    # 0 - NormScorer
    # 1 - KMeansScorer
    # 2 - DifferenceScorer
python -u station_trainer/darts_station_trainer.py \
  -s BRAZ \
  -f 0 \
  -si 2