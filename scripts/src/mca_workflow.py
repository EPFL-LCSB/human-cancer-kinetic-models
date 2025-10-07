"""
mca_workflow.py
---------------
This script performs metabolic control analysis (MCA) using a kinetic model and steady-state samples.
Paths and solver settings are read from config.ini for reproducibility and portability.
This version is set up for the WT (wild-type) case.
"""

import configparser
import os
import numpy as np
import pandas as pd
from pytfa.io.json import load_json_model
from skimpy.io.yaml import load_yaml_model
from skimpy.utils.namespace import QSSA
from skimpy.utils.tabdict import TabDict
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from scipy.sparse import *
import multiprocessing as mp
import os.path
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import seaborn as sns
from utils.run_kin_param_sample import sample_kin_param
from utils.get_fccs_for_reaction import get_flux_control_coefficients, get_fccs_parallel
from utils.remove_outliers import remove_outliers_parallel
from utils.check_convergence import check_convergence

LOWER_IX = 0
UPPER_IX = 1000
PHYSIOLOGY = 'WT'


# Read configuration from config.ini
config = configparser.ConfigParser()
config_path = os.path.join(os.path.dirname(__file__), 'config.ini')
config.read(config_path)

# Path to data and model from config, using base_dir
base_dir = config['paths']['base_dir']
path_to_kmodel = os.path.abspath(os.path.join(base_dir, config['paths'][f'path_to_kmodel_{PHYSIOLOGY}']))
path_to_tmodel = os.path.abspath(os.path.join(base_dir, config['paths'][f'path_to_tmodel_{PHYSIOLOGY}']))
path_to_samples = os.path.abspath(os.path.join(base_dir, config['paths'][f'path_to_samples_{PHYSIOLOGY}']))
path_to_lambda_values = os.path.abspath(os.path.join(base_dir, config['paths'][f'path_to_lambda_values_{PHYSIOLOGY}']))
path_to_param_output = os.path.abspath(os.path.join(base_dir, config['paths'][f'path_to_param_output_{PHYSIOLOGY}']))
path_to_fcc = os.path.abspath(os.path.join(base_dir, config['paths'][f'path_to_fcc_{PHYSIOLOGY}']))

# Scaling parameters from config
scaling_section = 'scaling'
CONCENTRATION_SCALING = float(config[scaling_section].get('CONCENTRATION_SCALING', 1e6))
TIME_SCALING = float(config[scaling_section].get('TIME_SCALING', 1))
DENSITY = float(config[scaling_section].get('DENSITY', 1200))
GDW_GWW_RATIO = float(config[scaling_section].get('GDW_GWW_RATIO', 0.25))
flux_scaling_factor = 1e-3 * (GDW_GWW_RATIO * DENSITY) * CONCENTRATION_SCALING / TIME_SCALING

# Load pytfa model
print('Loading TFA model from:', path_to_tmodel)
tmodel = load_json_model(path_to_tmodel)

print('Configuring solver...')
solver_section = 'solver_for_model_building'
solver_name = config[solver_section]['solver']
tmodel.solver = solver_name
tmodel.solver.configuration.tolerances.feasibility = float(config[solver_section]['feasibility_tolerance'])
tmodel.solver.configuration.tolerances.optimality = float(config[solver_section]['optimality_tolerance'])
tmodel.solver.configuration.tolerances.integrality = float(config[solver_section]['integrality_tolerance'])
tmodel.solver.problem.parameters.emphasis.numerical = int(config[solver_section]['numerical_emphasis'])
tmodel.solver.configuration.presolve = config.getboolean(solver_section, 'presolve')
tmodel.solver.problem.parameters.read.scale = int(config[solver_section]['read_scale'])

# Doubling time 26.1 hours from the biomass upper bound (ln(2)/0.026)
sol = tmodel.optimize()
MAX_EIGENVALUES = -3 * 1 / (
        np.log(2) / sol.objective_value)  # 3 times faster than the doubling time of the cell (~38.4 hours)
MIN_EIGENVALUES = -1 / (1e-6/3600)  # 1/1mus = 1/1e-6/60/60 1/h

# Load steady-state samples
print('Loading steady-state samples from:', path_to_samples)
samples = pd.read_csv(path_to_samples, header=0, index_col=0)

# Load the kinetic model and prepare for compiling of jacobian
ncpu = int(config['global'].get('ncpu', 1))
kmodel = load_yaml_model(path_to_kmodel)
kmodel.prepare()
print(f'Compiling Jacobian...')
kmodel.compile_jacobian(ncpu=ncpu)

# Compile with parameter elasticities
PARAMETER_FOR_MCA = 'vmax_forward'
parameter_list = TabDict([(k, p.symbol) for k, p in kmodel.parameters.items()
                          if p.name.startswith(PARAMETER_FOR_MCA)])

print('Compiling MCA...')
kmodel.compile_mca(sim_type=QSSA, ncpu=ncpu, parameter_list=parameter_list)

print('Kinetic model compiled successfully.')

# Read kinetic parameter sampler settings from config
kin_param_section = 'kin_param_sampler'
n_param_samples = int(config[kin_param_section].get('n_samples', 100))
only_stable = config[kin_param_section].get('only_stable', 'False').lower() == 'true'
min_max_eigenvalues = config[kin_param_section].get('min_max_eigenvalues', 'True').lower() == 'true'
print('Beginning kinetic parameter sampling in parallel...')
pool = mp.Pool(int(ncpu))
lambda_max_min = pool.starmap_async(sample_kin_param, [(i, samples.iloc[i,:]) for i in range(LOWER_IX, UPPER_IX)])
lambda_max_min = lambda_max_min.get()
pool.close()
pool.terminate()

print('Curating sampling results...')
lambda_min = {}
lambda_max = {}
for i in range(0, len(lambda_max_min)):
    lambda_max['sample ' + str(i)] = lambda_max_min[i][0]
    lambda_min['sample ' + str(i)] = lambda_max_min[i][1]

lambda_df = pd.DataFrame.from_dict(lambda_max, orient='index')
# Print some summary statistics
print('Summary statistics for lambda_max:')
print(lambda_df.describe())


print('Producing flux control coefficients...')
for ss in tqdm(range(LOWER_IX, UPPER_IX)):
    pool = mp.Pool(int(ncpu))
    results = pool.starmap_async(get_flux_control_coefficients, [(i,i+10 ,ss, samples, tmodel, kmodel, path_to_lambda_values, path_to_param_output, path_to_fcc) for i in range(0, 100, 10)])
    results = results.get()
    pool.close()
    pool.terminate()

# Get FCC matrix for biomass reaction
print('Getting FCC matrix for biomass reaction...')
biomass_fcc = get_fccs_parallel(range(LOWER_IX, UPPER_IX), reactions=['biomass'], path=path_to_fcc, n_cores=int(ncpu//2))

# Remove outliers with IQR
print('Removing outliers from FCC data...')
biomass_fcc_no_outliers = remove_outliers_parallel(biomass_fcc, n_jobs=-1)

# Check if the 3 first moments have converged
print('Checking convergence of FCC moments...')
converged = check_convergence(biomass_fcc_no_outliers, latest_ss_index=UPPER_IX-1, moment_number=3)

print("MCA workflow completed. Converged:", converged)