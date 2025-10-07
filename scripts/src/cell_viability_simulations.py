from pytfa.io.json import load_json_model
from skimpy.io.yaml import load_yaml_model
from skimpy.core.solution import ODESolutionPopulation
from skimpy.core.parameters import ParameterValuePopulation, \
    load_parameter_population
from skimpy.utils.namespace import QSSA
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import multiprocessing as mp
from skimpy.core.reactions import Reaction
import numpy as np
import configparser
import os
from utils.drug_ode_simulation import run_simulation_ic50, simulate_sample, ODESolution_ic50, produce_biomass_df, CellViabilitySolution
from utils.make_flux_fun_parallel import make_flux_fun_parallel
from utils.enzyme_degradation_class import make_enzymedegradation

TIME = np.linspace(0, 600, 5) # 20-30 times the doubling time of the cell
PHYSIOLOGY = 'WT'
TARGETS = ['TMDS']
TARGET_NAME = 'TMDS'

print('Running cell viability simulations for:', TARGET_NAME, 'and physiology:', PHYSIOLOGY)

# Scaling parameters
config = configparser.ConfigParser()
config_path = os.path.join(os.path.dirname(__file__), 'config.ini')
config.read(config_path)
CONCENTRATION_SCALING = float(config['scaling']['CONCENTRATION_SCALING'])  # 1 mol to 1 mumol
TIME_SCALING = float(config['scaling']['TIME_SCALING'])  # 1 hour to 1 min
DENSITY = float(config['scaling']['DENSITY'])  # g/L
GDW_GWW_RATIO = float(config['scaling']['GDW_GWW_RATIO'])  # Assumes 75% Water
flux_scaling_factor = 1e-3 * (GDW_GWW_RATIO * DENSITY) * CONCENTRATION_SCALING / TIME_SCALING

# NCPU
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
path_to_params = os.path.abspath(os.path.join(base_dir, config['paths'][f'path_to_param_output_{PHYSIOLOGY}']))
path_to_max_eig = os.path.abspath(os.path.join(base_dir, config['paths'][f'path_to_lambda_values_{PHYSIOLOGY}']))
path_to_stratified_samples = os.path.abspath(os.path.join(base_dir, config['paths'][f'path_to_stratified_samples_picked_{PHYSIOLOGY}']))
path_to_stratified_params = os.path.abspath(os.path.join(base_dir, config['paths'][f'path_to_stratified_params_{PHYSIOLOGY}']))
path_to_cell_viability_solutions = os.path.abspath(os.path.join(base_dir, config['paths'][f'path_to_cell_viability_solutions_{PHYSIOLOGY}']))

print('Loading TFA model from:', path_to_tmodel)
tmodel = load_json_model(path_to_tmodel)

# Load the samples
samples = pd.read_csv(path_to_samples, index_col=0, header=0)


print('Loading kinetic model from:', path_to_kmodel)
kmodel = load_yaml_model(path_to_kmodel)
kmodel.prepare(mca=False)
print('Compiling ODEs...')
kmodel.compile_ode(sim_type=QSSA, ncpu=ncpu)

# Load the represenative models
samples_picked = pd.read_csv(path_to_stratified_samples, index_col=0)
parameter_population = load_parameter_population(path_to_stratified_params)

# Load the exponential decay parameters
degradation_rate = float(config['drug_metabolism']['degradation_rate'])
final_concentration = float(config['drug_metabolism']['final_concentration'])

for enz_name in TARGETS:
    print(f'Adding exponential degradation ODE for enzyme: {enz_name}')
    # In this section we create a new ODE for the enzyme degradation that represents the drug effect
    enzyme = tmodel.reactions.get_by_id(enz_name)
    name = 'exponential_degradation_{}'.format(enzyme.id)
    enzyme_name = 'E_{}'.format(enzyme.id)
    kcat_name = 'kcat_forward_{}'.format(enzyme.id)
    vmax_name = 'vmax_forward_{}'.format(enzyme.id)

    ED = make_enzymedegradation(enzyme_name)
    metabolites = ED.Reactants(**{enzyme_name:enzyme_name})

    ED_RXN = Reaction(name=name,
                    mechanism=ED,
                    reactants=metabolites,
                    )

    parameters = ED.Parameters(a=final_concentration, k=degradation_rate)

    kmodel.add_reaction(ED_RXN)
    kmodel.parametrize_by_reaction({ED_RXN.name:parameters})

    # Retain the compartment information even though they will not be used for the ODE
    kmodel.reactants[enzyme_name].compartment =kmodel.reactants.atp_c.compartment # randomly chosen the atp_c compartment

    # Replace the vmax with the kcat
    kmodel.parameters[kcat_name].value = kmodel.parameters[vmax_name].value
    kmodel.parameters[vmax_name].value = None

    # Assign enzyme to the reaction
    kmodel.reactions[enzyme.id].enzyme = kmodel.reactants[enzyme_name]
    kmodel.reactions[enzyme.id].mechanism.enzyme = kmodel.reactants[enzyme_name]
    kmodel.reactions[enzyme.id].mechanism.reactants['enzyme'] = kmodel.reactants[enzyme_name]



print('Rebuilding and compiling kinetic model...')
kmodel._modified = True
kmodel.compile_ode(sim_type = QSSA, ncpu=ncpu)
print('Compiling flux function...')
calc_flux = make_flux_fun_parallel(kmodel, QSSA, path_to_so_file = './tmp_remove_me.so')


# Run one time so that the parallelization works
samples_to_simulate = list(parameter_population._index.keys())
simulate_sample(samples_to_simulate[0], samples, tmodel=tmodel, kmodel=kmodel, 
                parameter_population=parameter_population, CONCENTRATION_SCALING=CONCENTRATION_SCALING,
                 targets=TARGETS, TIME=np.linspace(0, 1, 10), time_limit=1800,
                 a_exponentrial_degradation=final_concentration, k_exponentrial_degradation=degradation_rate,
                 rtol=1e-6, atol=1e-6)

# List of parameter values
a_vals = [0.999, 0.99, 0.9, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05, 0.01, 0.001, 1e-5, 1e-9]
ix_values = [i for i in samples_to_simulate]

# Create a list of (ix, a) tuples
args = [(ix, a) for ix in ix_values for a in a_vals]

# Run simulations in parallel using multiprocessing.Pool
print('Running simulations in parallel...')
if __name__ == '__main__':
    def run_ic50_wrapper(arg):
        ix, a = arg
        filename = path_to_cell_viability_solutions.format(TARGET_NAME, ix, a)
        return run_simulation_ic50(
            (ix, a), filename, samples, tmodel, kmodel, parameter_population, CONCENTRATION_SCALING,
            TARGETS, TIME, time_limit, a, degradation_rate, rtol, atol
        )

    with mp.Pool(processes=int(ncpu//2)) as pool:
        ic50_solutions = list(pool.imap(run_ic50_wrapper, args))


print('Finished running simulations in parallel.')