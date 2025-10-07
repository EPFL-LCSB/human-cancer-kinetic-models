"""
run_kin_param_sample.py
----------------------
This module provides a function to sample kinetic parameters for a given steady-state sample.
Configuration and scaling parameters are read from config.ini for reproducibility.
"""

import configparser
import os
import numpy as np
import pandas as pd
from pytfa.io.json import load_json_model
from skimpy.io.yaml import load_yaml_model
from skimpy.sampling.simple_parameter_sampler import SimpleParameterSampler
from skimpy.analysis.oracle.load_pytfa_solution import load_fluxes, load_concentrations, load_equilibrium_constants
from skimpy.core.parameters import ParameterValuePopulation

# Read scaling values from the config file
config = configparser.ConfigParser()
config_path = os.path.join(os.path.dirname(__file__), 'config.ini')
config.read(config_path)
scaling_section = 'scaling'
CONCENTRATION_SCALING = float(config[scaling_section].get('CONCENTRATION_SCALING', 1e6))
TIME_SCALING = float(config[scaling_section].get('TIME_SCALING', 1))
DENSITY = float(config[scaling_section].get('DENSITY', 1200))
GDW_GWW_RATIO = float(config[scaling_section].get('GDW_GWW_RATIO', 0.25))
flux_scaling_factor = 1e-3 * (GDW_GWW_RATIO * DENSITY) * CONCENTRATION_SCALING / TIME_SCALING

def sample_kin_param(i, sample, tmodel_, kmodel_, 
               path_to_param_output, path_to_lambda_values, override=False,
               n_param_samples=100, only_stable=False, min_max_eigenvalues=True):
    """
    Samples kinetic parameters for a given steady-state sample and saves results.
    """
    output_path = path_to_param_output.format(i)
    lambda_path = path_to_lambda_values.format(i)
    
    if not override and os.path.exists(output_path) and os.path.exists(lambda_path):
        print(f"File {output_path} already exists. Skipping sample {i}.")
        return

    print('Producing kinetic models from sample:', i)
    sampler_params = SimpleParameterSampler.Parameters(n_samples=n_param_samples)
    sampler = SimpleParameterSampler(sampler_params)

    # Load fluxes and concentrations
    fluxes = load_fluxes(sample, tmodel_, kmodel_,
                         density=DENSITY,
                         ratio_gdw_gww=GDW_GWW_RATIO,
                         concentration_scaling=CONCENTRATION_SCALING,
                         time_scaling=TIME_SCALING)

    concentrations = load_concentrations(sample, tmodel_, kmodel_,
                                         concentration_scaling=CONCENTRATION_SCALING)

    # Fetch equilibrium constants
    load_equilibrium_constants(sample, tmodel_, kmodel_,
                               concentration_scaling=CONCENTRATION_SCALING,
                               in_place=True)
    kmodel_.logger.name = f"Kinetic model copy {i}"
    params, lamda_max, lamda_min = sampler.sample(kmodel_,
                                                  fluxes,
                                                  concentrations,
                                                  only_stable=only_stable,
                                                  min_max_eigenvalues=min_max_eigenvalues,
                                                  max_trials=1e6)
    params_population = ParameterValuePopulation(params, kmodel_)
    params_population.save(output_path)

    lambda_df = pd.DataFrame([lamda_max, lamda_min], index=['max_eig', 'min_eig'])
    lambda_df.T.to_csv(lambda_path)
    return [lamda_max, lamda_min]