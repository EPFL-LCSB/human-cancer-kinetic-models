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
from utils.drug_ode_simulation import simulate_sample, ODESolution_prior, ODESolution_post, FluxSolution, produce_flux_df
from utils.make_flux_fun_parallel import make_flux_fun_parallel
from utils.enzyme_degradation_class import make_enzymedegradation

TIME = np.linspace(0, 600, 1200) # 20-30 times the doubling time of the cell
PHYSIOLOGY = 'WT'
TARGETS = ['TMDS']
TARGET_NAME = 'TMDS'

print('Running drug simulations for:', TARGET_NAME, 'and physiology:', PHYSIOLOGY)

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
path_to_ode_conc_solutions = os.path.abspath(os.path.join(base_dir, config['paths'][f'path_to_ode_conc_solutions_{PHYSIOLOGY}']))
path_to_ode_flux_solutions = os.path.abspath(os.path.join(base_dir, config['paths'][f'path_to_ode_flux_solutions_{PHYSIOLOGY}']))


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

# Buffer function
def run_simulation(ix):
    return simulate_sample(ix, samples, tmodel=tmodel, kmodel=kmodel, 
                           parameter_population=parameter_population, CONCENTRATION_SCALING=CONCENTRATION_SCALING,
                           targets=TARGETS, TIME=TIME, time_limit=time_limit,
                           a_exponentrial_degradation=final_concentration, k_exponentrial_degradation=degradation_rate,
                           rtol=rtol, atol=atol)


print('Running ODE simulations in parallel...')
if __name__ == '__main__':
    with mp.Pool(processes=int(ncpu//2)) as pool:
        solutions = list(pool.imap(run_simulation, samples_to_simulate))


print('Saving ODE concentration solutions...')
sol = []
for i in tqdm(solutions):
    sol.append(ODESolution_prior(i[0][1], i[0][0]))

solutions = sol
final_res = ODESolutionPopulation(solutions)
final_res.data.to_csv(path_to_ode_conc_solutions.format(TARGET_NAME))


print('Loading ODE concentration solutions for post-processing...')
final_res = pd.read_csv(path_to_ode_conc_solutions.format(TARGET_NAME))

# In case the solutions need to be loaded again
solutions = []
for ix, sol_id in zip(samples_to_simulate, range(final_res.solution_id.max()+1)):
    ids = np.where(final_res.solution_id == sol_id)[0]
    sol = ODESolution_post(final_res.iloc[ids,2:], final_res.iloc[ids,1], ix)
    solutions.append(sol)

solutions_raw = solutions
solutions = []
for sol, ix in zip(solutions_raw, samples_to_simulate):
    if sol.time[-1] == TIME[-1]:
        solutions.append(sol)
    else:
        print('Solution {} did not converge in time ({})'.format(ix, sol.time[-1]))


# Function to initialize the pool
def init_pool(kmodel):
    global kmodel_
    kmodel_ = kmodel


print('Calculating transient fluxes...')
if __name__ == '__main__':
    with mp.Pool(processes=int(ncpu//10), initializer=init_pool, initargs=(kmodel,)) as pool:
        results = list(tqdm(pool.imap(lambda res: produce_flux_df(res, kmodel, parameter_population, TARGETS, calc_flux), solutions), total=len(solutions)))

    fluxes_dict = {ix: df for ix, df in results}


print('Saving flux solutions...')
flux_solutions = []
for ix, df in fluxes_dict.items():
    flux_solutions.append(FluxSolution(df, TIME, ix))

# Save the flux solutions to csv file 
sol_cols = list(flux_solutions[0].fluxes.columns)
total_flux_df = pd.DataFrame(columns=['model_ix','time'] + sol_cols)
for i, sol in enumerate(flux_solutions):
    if i == 0:
        continue
    sol_df = pd.DataFrame(sol.fluxes)
    sol_df['time'] = sol.time
    sol_df['model_ix'] = sol.model_ix
    total_flux_df = total_flux_df.append(sol_df)

total_flux_df.to_csv(path_to_ode_flux_solutions.format(TARGET_NAME))
print('All results saved successfully.')