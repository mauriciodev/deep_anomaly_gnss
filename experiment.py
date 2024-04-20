"""
NNI hyperparameter optimization for TimesNet on GNSS data.

Check the online tutorial for details:
https://nni.readthedocs.io/en/stable/tutorials/hpo_quickstart_pytorch/main.html
"""

import signal
from nni.experiment import Experiment

params = {
    'seq_len':{"_type": "choice", "_value": [100, 50, 25, 10]},
    'lr':{"_type": "loguniform", "_value": [1e-4, 1e-2]},
    'epochs':{"_type": "choice", "_value": [5, 10, 15]},
    'batch_size':{"_type": "choice", "_value": [32, 64, 128]},
    'epoch_steps':{"_type": "choice", "_value": [10, 20, 30, 40, 50]},
    'e_layers':{"_type": "choice", "_value": [2, 3, 4]},
    'd_model':{"_type": "choice", "_value": [64, 128, 256]},
    'd_ff':{"_type": "choice", "_value": [64, 128, 256]},
    'dropout':{"_type": "uniform", "_value": [0.1, 0.3]},
    'top_k':{"_type": "choice", "_value": [3, 5, 10]},
    'num_kernels':{"_type": "choice", "_value": [3, 6, 12]},
    'percentile':{"_type": "uniform", "_value": [96, 99]},
}

# Configure experiment
experiment = Experiment('local')
experiment.config.experiment_name = 'TimesNet Hyper Tune'
experiment.config.search_space = params
experiment.config.trial_command = 'python timesnet.py'
experiment.config.trial_code_directory = '.'
#experiment.config.trial_gpu_number = 0
experiment.config.trial_concurrency = 1
experiment.config.max_trial_number = 10
experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args['optimize_mode'] = 'maximize'
#experiment.config.training_service.use_active_gpu = False

# Run it!
experiment.run(port=8080, wait_completion=False)

print('Experiment is running. Press Ctrl-C to quit.')
signal.pause()
