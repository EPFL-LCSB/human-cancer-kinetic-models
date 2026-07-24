"""
smoke_test.py
-------------
This script performs a lightweight check of the repository inputs.
It loads one provided steady-state model file, reads a precomputed MCA result,
applies a small perturbation estimate, and verifies that the expected output is
finite and non-empty.
"""

import configparser
import json
import os

import numpy as np
import pandas as pd

PHYSIOLOGY = 'WT'
METABOLITE = 'biomass'
PERTURBATION_FRACTION = 0.10


# Read configuration from config.ini
config = configparser.ConfigParser()
config_path = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'src', 'config.ini')
config.read(config_path)

# Path to data and model from config, using base_dir
base_dir = config['paths']['base_dir']
path_to_tmodel = os.path.abspath(os.path.join(base_dir, config['paths'][f'path_to_tmodel_{PHYSIOLOGY}']))
path_to_cc_df = os.path.abspath(os.path.join(base_dir, config['paths'][f'path_to_cc_df_{PHYSIOLOGY}']))


def require_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)


# Load a provided model file without rerunning optimization or sampling
print('Loading TFA model file from:', path_to_tmodel)
require_file(path_to_tmodel)
with open(path_to_tmodel, 'r') as handle:
    tmodel_data = json.load(handle)

if 'metabolites' not in tmodel_data or 'reactions' not in tmodel_data:
    raise ValueError('Model file does not contain metabolites and reactions.')

if len(tmodel_data['metabolites']) == 0 or len(tmodel_data['reactions']) == 0:
    raise ValueError('Model file contains no metabolites or reactions.')


# Load a stored MCA result for a small perturbation-style calculation
path_to_biomass_cc = path_to_cc_df.format(METABOLITE)
print('Loading MCA matrix from:', path_to_biomass_cc)
require_file(path_to_biomass_cc)
mca_data = pd.read_csv(path_to_biomass_cc, header=0, index_col=0)
mca_data = mca_data.apply(pd.to_numeric, errors='coerce')

if mca_data.empty:
    raise ValueError('MCA matrix is empty.')

mean_coefficients = mca_data.mean(axis=1).dropna()
if mean_coefficients.empty:
    raise ValueError('No numeric MCA coefficients were found.')

top_parameter = mean_coefficients.abs().sort_values(ascending=False).index[0]
top_coefficient = mean_coefficients.loc[top_parameter]

# First-order MCA approximation for a small enzyme perturbation
predicted_relative_change = top_coefficient * PERTURBATION_FRACTION

smoke_test_output = pd.DataFrame([{
    'physiology': PHYSIOLOGY,
    'metabolite': METABOLITE,
    'parameter': top_parameter,
    'mean_control_coefficient': top_coefficient,
    'perturbation_fraction': PERTURBATION_FRACTION,
    'predicted_relative_change': predicted_relative_change,
}])

expected_columns = [
    'physiology',
    'metabolite',
    'parameter',
    'mean_control_coefficient',
    'perturbation_fraction',
    'predicted_relative_change',
]

if list(smoke_test_output.columns) != expected_columns:
    raise ValueError('Smoke test output columns do not match the expected format.')

if smoke_test_output.shape != (1, len(expected_columns)):
    raise ValueError('Smoke test output has an unexpected shape.')

if not np.isfinite(smoke_test_output['predicted_relative_change'].iloc[0]):
    raise ValueError('Predicted perturbation output is not finite.')

print('Model metabolites:', len(tmodel_data['metabolites']))
print('Model reactions:', len(tmodel_data['reactions']))
print('MCA matrix shape:', mca_data.shape)
print('Smoke test output:')
print(smoke_test_output.to_string(index=False))
print('Smoke test completed successfully.')
