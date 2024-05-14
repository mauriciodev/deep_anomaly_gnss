"""
NNI hyperparameter optimization for TimesNet on GNSS data.

Check the online tutorial for details:
https://nni.readthedocs.io/en/stable/tutorials/hpo_quickstart_pytorch/main.html
"""

import signal
from nni.experiment import Experiment
import argparse

parser = argparse.ArgumentParser(prog='experiment')
parser.add_argument(
    '-s',
    '-station',
    help='Station name',
    default='BRAZ' # positional argument
)
parser.add_argument(
    '-c',
    '-concurrency',
    help='Number of parallel experiments',
    type=int,
    default=1 # positional argument
)
parsed_args =parser.parse_args()
station = parsed_args.s

params = {
    'seq_len':{"_type": "choice", "_value": [10, 20]},
    'e_layers':{"_type": "choice", "_value": [2, 3]},
    'd_model':{"_type": "choice", "_value": [32, 64, 128]},
    'd_ff':{"_type": "choice", "_value": [32, 64, 128]},
    'dropout':{"_type": "quniform", "_value": [0.1, 0.5, 0.05]},
    'top_k':{"_type": "choice", "_value": [3, 4, 5]},
}

# Configure experiment
experiment = Experiment('local')
experiment.config.experiment_name = f'TimesNet Hyper Tune {station}'
experiment.config.search_space = params
experiment.config.trial_command = f"python nni_experiment/nni_timesnet_experiment.py -s {station}"
experiment.config.trial_code_directory = '.'
#experiment.config.trial_gpu_number = 1
experiment.config.trial_concurrency = parsed_args.c
experiment.config.max_trial_number = 50
experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args['optimize_mode'] = 'minimize'
experiment.config.training_service.use_active_gpu = True

# Run it!
experiment.run(port=8080, wait_completion=True)

print('Experiment is running. Press Ctrl-C to quit.')
#signal.pause()
