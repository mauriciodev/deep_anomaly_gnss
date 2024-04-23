"""
NNI hyperparameter optimization for TimesNet on GNSS data.

Check the online tutorial for details:
https://nni.readthedocs.io/en/stable/tutorials/hpo_quickstart_pytorch/main.html
"""

import signal
from nni.experiment import Experiment

params = {
    'seq_len':{"_type": "choice", "_value": [20]},
    'lr':{"_type": "loguniform", "_value": [1e-4, 1e-2]},
    #'epochs':{"_type": "choice", "_value": [5, 10, 15]},
    'batch_size':{"_type": "choice", "_value": [64]},
    'epoch_steps':{"_type": "choice", "_value": [40]},
    'e_layers':{"_type": "choice", "_value": [2, 3]},
    'd_model':{"_type": "choice", "_value": [256]},
    'd_ff':{"_type": "choice", "_value": [64]},
    'dropout':{"_type": "uniform", "_value": [0.1, 0.2]},
    'top_k':{"_type": "choice", "_value": [6, 7]},
    'num_kernels':{"_type": "choice", "_value": [6]},
    'percentile':{"_type": "uniform", "_value": [99, 99.5]},
    'patience':{"_type": "choice", "_value": [3, 5, 7]},
    'delta':{"_type": "uniform", "_value": [1e-6, 5e-6]},
}

# Configure experiment
experiment = Experiment('local')
experiment.config.experiment_name = 'TimesNet Hyper Tune'
experiment.config.search_space = params
experiment.config.trial_command = 'python gnss_ad_timesnet.py'
experiment.config.trial_code_directory = '.'
#experiment.config.trial_gpu_number = 1
experiment.config.trial_concurrency = 1
experiment.config.max_trial_number = 50
experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args['optimize_mode'] = 'maximize'
#experiment.config.training_service.use_active_gpu = True

# Run it!
experiment.run(port=8080, wait_completion=False)

print('Experiment is running. Press Ctrl-C to quit.')
signal.pause()