# Run the full process since the preprocessing
bash scripts/preprocess.sh
bash scripts/nni_experiments.sh
bash scripts/timesnet.sh
bash scripts/darts.sh