from nni.experiment import Experiment
experiment = Experiment('local')

experiment.config.trial_command = 'python deepOD.py'
experiment.config.trial_code_directory = '.'

search_space = {
    'seq_len': {'_type': 'choice', '_value': [10, 20, 30]},
    'epochs':  {'_type': 'choice', '_value': [10, 20, 30]},
}

experiment.config.search_space = search_space

experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args['optimize_mode'] = 'maximize'

experiment.config.max_trial_number = 10
experiment.config.trial_concurrency = 1

experiment.run(8080)
input('Press enter to quit')
experiment.stop()
