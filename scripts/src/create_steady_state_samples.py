"""
create_steady_state_samples.py
-----------------------------
This script generates steady-state samples from a TFA model using the Generalized ACHR Sampler.
It loads the TFA model, removes integer variables, performs sampling, and saves the samples to a CSV file.
"""

import configparser
import os

from pytfa.io.json import load_json_model
from pytfa.analysis import GeneralizedACHRSampler
from pytfa.optim import strip_from_integer_variables
from skimpy.analysis.oracle.minimum_fluxes import MinFLuxVariable

PHYSIOLOGY = 'WT'

# Read configuration from config.ini
config = configparser.ConfigParser()
config_path = os.path.join(os.path.dirname(__file__), 'config.ini')
config.read(config_path)

# Path to data and model from config, using base_dir
base_dir = config['paths']['base_dir']
path_to_tmodel = os.path.abspath(os.path.join(base_dir, config['paths'][f'path_to_tmodel_{PHYSIOLOGY}']))
path_to_samples = os.path.abspath(os.path.join(base_dir, config['paths'][f'path_to_samples_{PHYSIOLOGY}']))

# Read sampler parameters from config
sampler_section = 'steady_state_sampler'
n_samples = int(config[sampler_section].get('n_samples', 5000))
thinning = int(config[sampler_section].get('thinning', 500))
seed = int(config[sampler_section].get('seed', 123))

# Load TFA model
print('Loading TFA model from:', path_to_tmodel)
tmodel = load_json_model(path_to_tmodel)

# Remove integer variables in preparation for sampling
print('Stripping integer variables from model...')
continuous_model = strip_from_integer_variables(tmodel)

# Initialize sampler
print(f'Initializing Generalized ACHR Sampler (thinning={thinning}, seed={seed})...')
sampler = GeneralizedACHRSampler(continuous_model, thinning=thinning, seed=seed)

# Perform sampling
print(f'Sampling {n_samples} steady-state concentrations...')
samples = sampler.sample(n_samples, fluxes=False)

# Save samples to CSV
print('Saving samples to:', path_to_samples)
samples.to_csv(path_to_samples)