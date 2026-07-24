"""
ode_smoke_test.py
-----------------
This script performs a lightweight ODE smoke test using one precomputed WT
steady-state sample and one precomputed kinetic parameter set.
It compiles the ODE system, reduces the TMDS vmax to 10% of its original value,
integrates the model for 10 model hours, and verifies that the output is finite
and non-empty.
"""

import configparser
import os
import time

import numpy as np
import pandas as pd
from pytfa.io.json import load_json_model
from skimpy.analysis.oracle.load_pytfa_solution import load_concentrations, \
    load_equilibrium_constants
from skimpy.core.parameters import load_parameter_population
from skimpy.core.solution import ODESolution
from skimpy.io.yaml import load_yaml_model
from skimpy.utils.namespace import QSSA
from skimpy.utils.tabdict import TabDict

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))
from utils.drug_ode_simulation import solver_ode

PHYSIOLOGY = 'WT'
TARGET = 'TMDS'
PERTURBATION_FRACTION = 0.10
TIME = np.linspace(0, 10, 21)


def require_file(path, label='file'):
    if not os.path.exists(path):
        raise FileNotFoundError('Required {} not found: {}'.format(label, path))


def select_model_index(parameter_population, selected_steady_states):
    for model_ix in parameter_population._index.keys():
        tfa_id, _ = model_ix.split(',')
        if int(tfa_id) in selected_steady_states:
            return model_ix
    raise ValueError('No kinetic parameter sample matches the selected steady states.')


# Read configuration from config.ini
config = configparser.ConfigParser()
config_path = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'src', 'config.ini')
config.read(config_path)

# Scaling parameters from config.ini
CONCENTRATION_SCALING = float(config['scaling']['CONCENTRATION_SCALING'])  # 1 mol to 1 mumol
TIME_SCALING = float(config['scaling']['TIME_SCALING'])  # 1 hour to 1 min
DENSITY = float(config['scaling']['DENSITY'])  # g/L
GDW_GWW_RATIO = float(config['scaling']['GDW_GWW_RATIO'])  # Assumes 75% Water

# Use the configured CPU count for ODE compilation
ncpu = int(config['global']['ncpu'])

# ODE parameters
time_limit = float(config['drug_metabolism']['time_limit'])
rtol = float(config['drug_metabolism']['rtol'])
atol = float(config['drug_metabolism']['atol'])

# Paths from config.ini using PHYSIOLOGY variable
base_dir = config['paths']['base_dir']
path_to_kmodel = os.path.abspath(os.path.join(base_dir, config['paths'][f'path_to_kmodel_{PHYSIOLOGY}']))
path_to_tmodel = os.path.abspath(os.path.join(base_dir, config['paths'][f'path_to_tmodel_{PHYSIOLOGY}']))
path_to_samples = os.path.abspath(os.path.join(base_dir, config['paths'][f'path_to_samples_{PHYSIOLOGY}']))
path_to_stratified_samples = os.path.abspath(os.path.join(base_dir, config['paths'][f'path_to_stratified_samples_{PHYSIOLOGY}']))
path_to_stratified_params = os.path.abspath(os.path.join(base_dir, config['paths'][f'path_to_stratified_params_{PHYSIOLOGY}']))


print('Using ncpu:', ncpu)

require_file(path_to_tmodel, 'WT TFA model')
require_file(path_to_samples, 'WT steady-state samples')
require_file(path_to_stratified_samples, 'selected WT steady-state samples')
require_file(path_to_stratified_params, 'selected WT kinetic parameter subset')
require_file(path_to_kmodel, 'WT kinetic model')

print('Loading TFA model from:', path_to_tmodel)
tmodel = load_json_model(path_to_tmodel)

print('Loading steady-state samples from:', path_to_samples)
samples = pd.read_csv(path_to_samples, index_col=0, header=0)

print('Loading selected steady-state samples from:', path_to_stratified_samples)
samples_picked = pd.read_csv(path_to_stratified_samples, index_col=0)
selected_steady_states = set(samples_picked.index.astype(int))

print('Loading kinetic parameters from:', path_to_stratified_params)
parameter_population = load_parameter_population(path_to_stratified_params)
model_ix = select_model_index(parameter_population, selected_steady_states)
tfa_id, parameter_id = model_ix.split(',')
tfa_id = int(tfa_id)

print('Selected model index:', model_ix)
print('Selected steady-state id:', tfa_id)
print('Selected kinetic parameter id:', parameter_id)

print('Loading kinetic model from:', path_to_kmodel)
kmodel = load_yaml_model(path_to_kmodel)
kmodel.prepare(mca=False)

print('Compiling ODEs...')
kmodel.compile_ode(sim_type=QSSA, ncpu=ncpu)

# Load the selected steady state and parameter set
tfa_sample = samples.loc[tfa_id]
parameter_set = parameter_population[model_ix]
kmodel.parameters = parameter_set

load_equilibrium_constants(tfa_sample, tmodel, kmodel,
                           concentration_scaling=CONCENTRATION_SCALING,
                           in_place=True)

init_concentrations = load_concentrations(tfa_sample, tmodel, kmodel,
                                          concentration_scaling=CONCENTRATION_SCALING)

# Reduce the target vmax directly before simulation
vmax_name = 'vmax_forward_{}'.format(TARGET)
if vmax_name not in kmodel.parameters:
    raise KeyError('Parameter not found: {}'.format(vmax_name))

original_vmax = kmodel.parameters[vmax_name].value
kmodel.parameters[vmax_name].value = PERTURBATION_FRACTION * original_vmax
perturbed_vmax = kmodel.parameters[vmax_name].value

print('{} changed from {} to {}'.format(vmax_name, original_vmax, perturbed_vmax))


def rootfn(t, y, g, user_data):
    t_0 = user_data['time_0']
    t_max = user_data['max_time']
    t = time.time()
    if (t - t_0) >= t_max:
        g[0] = 0
        print('Did not converge in time')
    else:
        g[0] = 1


user_data = {'time_0': time.time(),
             'max_time': time_limit}

# Order initial concentrations according to the compiled ODE variables
kmodel.initial_conditions = TabDict([(k, v) for k, v in init_concentrations.iteritems()])
ordered_initial_conditions = [kmodel.initial_conditions[variable]
                              for variable in kmodel.variables]

print('Solving ODE system for {} model hours...'.format(TIME[-1]))
solver = solver_ode(kmodel,
                    TIME,
                    solver_type='cvode',
                    rtol=rtol,
                    atol=atol,
                    max_steps=1e6,
                    rootfn=rootfn,
                    nr_rootfns=1,
                    user_data=user_data)

solver.init_step(0, ordered_initial_conditions)
this_sol_qssa = solver.solve(TIME, ordered_initial_conditions)
ode_solution = ODESolution(kmodel, this_sol_qssa)
concentrations = ode_solution.concentrations

if concentrations.empty:
    raise ValueError('ODE solution is empty.')

if len(ode_solution.time) != len(TIME):
    raise ValueError('ODE solution has an unexpected number of time points.')

if ode_solution.time[-1] != TIME[-1]:
    raise ValueError('ODE solution did not reach the final time point.')

numeric_concentrations = concentrations.apply(pd.to_numeric, errors='coerce')
if not np.isfinite(numeric_concentrations.values).all():
    raise ValueError('ODE solution contains non-finite concentrations.')

if not np.isfinite(float(perturbed_vmax)):
    raise ValueError('Perturbed vmax is not finite.')

print('ODE solution shape:', concentrations.shape)
print('Final simulation time:', ode_solution.time[-1])
print('ODE smoke test completed successfully.')
