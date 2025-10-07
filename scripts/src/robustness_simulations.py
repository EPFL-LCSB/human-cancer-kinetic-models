#This code is to assess the stability of the kinetic model 
# used to represent the MUT state of the kinetic model in the NRA formulation

import numpy as np
import pandas as pd

from pytfa.io.json import load_json_model
from pytfa.optim.constraints import *

from skimpy.io.yaml import load_yaml_model
from skimpy.core.parameters import load_parameter_population
from skimpy.utils.namespace import NET, QSSA

import os
import time as T
from pytfa.io.json import load_json_model
from skimpy.io.yaml import load_yaml_model
from skimpy.core.parameters import load_parameter_population
from skimpy.utils.namespace import QSSA
from skimpy.utils.tabdict import TabDict
import os
import numpy as np
import pandas as pd
import configparser
from utils.robustness_test_simulation import perturb_ode_solver


config = configparser.ConfigParser()
config.read(os.path.abspath(os.path.join(os.path.dirname(__file__), 'config.ini')))

# Cellular parameters from config
CONCENTRATION_SCALING = float(config['scaling']['CONCENTRATION_SCALING'])
TIME_SCALING = float(config['scaling']['TIME_SCALING'])
DENSITY = float(config['scaling']['DENSITY'])
GDW_GWW_RATIO = float(config['scaling']['GDW_GWW_RATIO'])
flux_scaling_factor = 1e-3 * (GDW_GWW_RATIO * DENSITY) * CONCENTRATION_SCALING / TIME_SCALING

ncpu = config['global']['ncpu']

# Paths from config
base_dir = config['paths']['base_dir']
path_to_tmodel = os.path.join(base_dir, config['paths']['path_to_tmodel_MUT'])
path_to_kmodel = os.path.join(base_dir, config['paths']['path_to_kmodel_MUT'])
path_to_samples = os.path.join(base_dir, config['paths']['path_to_samples_MUT'])
path_to_param_output_MUT = os.path.join(base_dir, config['paths']['path_to_param_output_MUT'])
path_to_robustness_output = os.path.join(base_dir, config['paths']['path_to_robustness_output'])

# Load pytfa model
tmodel = load_json_model(path_to_tmodel)
CPLEX = 'optlang-cplex'
tmodel.solver = CPLEX

tmodel.solver.configuration.tolerances.feasibility = float(config['solver_for_model_building']['feasibility_tolerance'])
tmodel.solver.configuration.tolerances.optimality = float(config['solver_for_model_building']['optimality_tolerance'])
tmodel.solver.configuration.tolerances.integrality = float(config['solver_for_model_building']['integrality_tolerance'])
doubling_time = np.log(2)/tmodel.reactions.biomass.lower_bound


# Load tfa samples
samples = pd.read_csv(path_to_samples, header=0, index_col=0)

tfa_id = 17 
sample = samples.loc[tfa_id]
doubling_time = np.log(2)/tmodel.reactions.biomass.lower_bound
TIME = np.logspace(-9, np.log10(doubling_time+0.1), 1000)  # Iintegration time just above the doubling time of 26.1 hr

# Compile ODE-Functions
kmodel = load_yaml_model(path_to_kmodel)
kmodel.prepare()
kmodel.compile_ode(sim_type=QSSA, ncpu=ncpu)

parameter_population = load_parameter_population(path_to_param_output_MUT.format(tfa_id))
tfa_ix = 17
tfa_sample = samples.iloc[tfa_ix]
parameter_set = parameter_population['0']

# Testing and initializing some subclasses of the function
N_SAMPLES = 1
RELATIVE_CHANGE = 0.001
TOTAL_TIME_STEPS = 10
TIME = np.logspace(-9, np.log10(doubling_time), TOTAL_TIME_STEPS)
TIME = np.append(TIME, np.logspace(np.log10(doubling_time+0.1), np.log10(doubling_time*3), 10))
ll = perturb_ode_solver('0', tmodel, kmodel, samples, parameter_population, tfa_id, path_to_robustness_output,
                       RELATIVE_CHANGE = RELATIVE_CHANGE, N_SAMPLES = N_SAMPLES, TIME = TIME,
                       TOTAL_TIME_STEPS = TOTAL_TIME_STEPS,
                       rtol= 1e-9, atol=1e-9)
# This is the real run
N_SAMPLES = 100
RELATIVE_CHANGE = 0.1
TOTAL_TIME_STEPS = 10
TIME = np.logspace(-9, np.log10(doubling_time), TOTAL_TIME_STEPS)
TIME = np.append(TIME, np.logspace(np.log10(doubling_time+0.1), np.log10(doubling_time*3), 10))

ll = perturb_ode_solver('0', tmodel, kmodel, samples, parameter_population, tfa_id, path_to_robustness_output,
                       RELATIVE_CHANGE = RELATIVE_CHANGE, N_SAMPLES = N_SAMPLES, TIME = TIME,
                       TOTAL_TIME_STEPS = TOTAL_TIME_STEPS,
                       rtol= 1e-9, atol=1e-9)

ll, res, steady_state_values = ll

print('All results should be fast and stable (code number: 1)')
print(res)

print('Robustness tests finished successfully')