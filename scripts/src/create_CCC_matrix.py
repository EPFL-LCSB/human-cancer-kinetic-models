from pytfa.io.json import load_json_model
from skimpy.io.yaml import load_yaml_model
from skimpy.analysis.ode import sample_initial_concentrations
from skimpy.core.solution import ODESolutionPopulation
from skimpy.analysis.mca.utils import get_dep_indep_vars_from_basis
from skimpy.utils.general import get_stoichiometry
from skimpy.utils.tensor import Tensor
from skimpy.analysis.ode.utils import make_flux_fun
from skimpy.utils.namespace import QSSA
from skimpy.utils.tabdict import TabDict
from skimpy.sampling.simple_parameter_sampler import SimpleParameterSampler
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import pickle
from scipy.sparse import *
import multiprocessing as mp
from skimpy.analysis.oracle.load_pytfa_solution import load_fluxes, \
    load_concentrations, load_equilibrium_constants
from skimpy.core.parameters import ParameterValuePopulation, \
    load_parameter_population
import os.path
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import seaborn as sns
import sys
sys.path.append('../')
from utils.remove_outliers import remove_outliers_parallel
import configparser
import os
from utils.get_cccs_for_metabolite import get_conc_control_coefficients, get_cccs_parallel

LOWER_IX = 0
UPPER_IX = 1000
PHYSIOLOGY = 'WT'
METABOLITES = ['atp_c', 'adp_c', 'amp_c']


# Read configuration from config.ini (single instance, robust path)
config = configparser.ConfigParser()
config_path = os.path.join(os.path.dirname(__file__), 'config.ini')
config.read(config_path)

# Scaling parameters from config.ini
CONCENTRATION_SCALING = float(config['scaling']['CONCENTRATION_SCALING'])  # 1 mol to 1 mmol
TIME_SCALING = float(config['scaling']['TIME_SCALING'])  # 1 hour
DENSITY = float(config['scaling']['DENSITY'])  # g/L
GDW_GWW_RATIO = float(config['scaling']['GDW_GWW_RATIO'])  # Assumes 75% Water

flux_scaling_factor = 1e-3 * (GDW_GWW_RATIO * DENSITY) * CONCENTRATION_SCALING / TIME_SCALING

# Path to data and model from config, using base_dir
base_dir = config['paths']['base_dir']
path_to_kmodel = os.path.abspath(os.path.join(base_dir, config['paths'][f'path_to_kmodel_{PHYSIOLOGY}']))
path_to_tmodel = os.path.abspath(os.path.join(base_dir, config['paths'][f'path_to_tmodel_{PHYSIOLOGY}']))
path_to_samples = os.path.abspath(os.path.join(base_dir, config['paths'][f'path_to_samples_{PHYSIOLOGY}']))
path_to_lambda_values = os.path.abspath(os.path.join(base_dir, config['paths'][f'path_to_lambda_values_{PHYSIOLOGY}']))
path_to_param_output = os.path.abspath(os.path.join(base_dir, config['paths'][f'path_to_param_output_{PHYSIOLOGY}']))
path_to_ccc = os.path.abspath(os.path.join(base_dir, config['paths'][f'path_to_ccc_{PHYSIOLOGY}']))
path_to_cc_df = os.path.abspath(os.path.join(base_dir, config['paths'][f'path_to_cc_df_{PHYSIOLOGY}']))

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

# Doubling time
sol = tmodel.optimize()
MAX_EIGENVALUES = -3 * 1 / (
        np.log(2) / sol.objective_value)

# Load steady-state samples
samples = pd.read_csv(path_to_samples, header=0, index_col=0)

# Load the kinetic model and prepare for compilinig of jacobian
kmodel = load_yaml_model(path_to_kmodel)
kmodel.prepare()

# Compile with parameter elasticities
PARAMETER_FOR_MCA = 'vmax_forward'
parameter_list = TabDict([(k, p.symbol) for k, p in kmodel.parameters.items()
                          if p.name.startswith(PARAMETER_FOR_MCA)])

# Load ncpu from config.ini [global] section
ncpu = int(config['global']['ncpu'])
kmodel.compile_mca(sim_type=QSSA, ncpu=ncpu, parameter_list=parameter_list)

print('Kinetic model compiled successfully.')

print('Producing concentration control coefficients...')
for ss in tqdm(range(LOWER_IX, UPPER_IX)):
    pool = mp.Pool(int(ncpu))
    results = pool.starmap_async(get_conc_control_coefficients, [(i,i+10 ,ss, samples, tmodel, kmodel, path_to_lambda_values, path_to_param_output, path_to_ccc) for i in range(0, 100, 10)])
    results = results.get()
    pool.close()
    pool.terminate()

# Get CCC matrix for selected metabolites
met_ccc = get_cccs_parallel(range(LOWER_IX, UPPER_IX), metabolites=METABOLITES, path=path_to_ccc, n_cores=int(ncpu//2))

# Remove outliers with IQR
print('Removing outliers from CCC data...')
for met in METABOLITES:
    met_ccc[met] = remove_outliers_parallel(met_ccc[met])

# Save the CCC data
print('Saving CCC data...')
for met in METABOLITES:
    met_ccc[met].to_csv(path_to_cc_df.format(met))