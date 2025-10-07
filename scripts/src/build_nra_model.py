import numpy as np
import pandas as pd

from pytfa.io.json import load_json_model
from pytfa.optim.constraints import *

import sys
sys.path.append("../../../../NRAplus/NRAplus") # Adds higher directory to python modules path.

from skimpy.io.yaml import load_yaml_model
from skimpy.analysis.oracle.load_pytfa_solution import load_fluxes, load_concentrations, load_equilibrium_constants
from skimpy.core.parameters import load_parameter_population
from skimpy.core.solution import ODESolutionPopulation
from skimpy.utils.tabdict import TabDict
from skimpy.utils.namespace import NET, QSSA
from skimpy.simulations.reactor import make_batch_reactor

from utils.NRAplus_parallel import NRAmodel_parallel, prepare_metabolites, prepare_reactions, add_vars_cons_from_dict
from pytfa.io.viz import get_reaction_data
from utils.nra_save_custom_json import save_json_nra_model

from sklearn.decomposition import PCA

import orjson
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

import os
import glob
import time as T
import matplotlib.pyplot as plt


# Cellular parameters

# Load parameters and paths from config.ini
import configparser

config = configparser.ConfigParser()
config.read(os.path.abspath(os.path.join(os.path.dirname(__file__), 'config.ini')))

ncpu = int(config['global']['ncpu'])

# Cellular parameters from config
CONCENTRATION_SCALING = float(config['scaling']['CONCENTRATION_SCALING'])
TIME_SCALING = float(config['scaling']['TIME_SCALING'])
DENSITY = float(config['scaling']['DENSITY'])
GDW_GWW_RATIO = float(config['scaling']['GDW_GWW_RATIO'])
flux_scaling_factor = 1e-3 * (GDW_GWW_RATIO * DENSITY) * CONCENTRATION_SCALING / TIME_SCALING

# Paths from config
base_dir = config['paths']['base_dir']
path_to_tmodel_WT = os.path.join(base_dir, config['paths']['path_to_tmodel_WT'])
path_to_tmodel_MUT = os.path.join(base_dir, config['paths']['path_to_tmodel_MUT'])
path_to_kmodel_MUT = os.path.join(base_dir, config['paths']['path_to_kmodel_MUT'])
path_to_tfa_samples_MUT = os.path.join(base_dir, config['paths']['path_to_samples_MUT'])
path_to_tfa_samples_WT = os.path.join(base_dir, config['paths']['path_to_samples_WT'])
path_to_significant_gene_expression_data = os.path.join(base_dir, config['paths']['path_to_significant_gene_expression_data'])
path_to_param_output_MUT = os.path.join(base_dir, config['paths']['path_to_param_output_MUT'])
path_to_nra_model = os.path.join(base_dir, config['paths']['path_to_nra_model'])

rxn_ratios = pd.read_csv(path_to_significant_gene_expression_data, index_col=0)

# Load models
tmodel_MUT = load_json_model(path_to_tmodel_MUT)
tmodel_WT = load_json_model(path_to_tmodel_WT)

# Find which reactions that had a significant fold change are transport reactions
significant_transport_rxns = []
ll = []
for i in rxn_ratios.index:
    rxn = tmodel_MUT.reactions.get_by_id(i)
    if "Transport" in rxn.subsystem:
        # print(rxn, '\n', rxn.subsystem)
        significant_transport_rxns.append(rxn.id)
    else:
        if rxn.subsystem not in ll:
            ll.append(rxn.subsystem)

inactive_transport_rxns = []
for rxn in tmodel_MUT.reactions:
    if "Transport" in rxn.subsystem and rxn.id not in significant_transport_rxns:
        inactive_transport_rxns.append('vmax_forward_' + rxn.id)


'''
Model and simulation initiation + preparation
'''
# Load models
kmodel_MUT = load_yaml_model(path_to_kmodel_MUT)

# Get the list of parameters for which we want control coefficients
parameter_list = TabDict([(k, p.symbol) for k, p in kmodel_MUT.parameters.items()
                          if p.name.startswith('vmax_forward')
                          and str(p.symbol) not in inactive_transport_rxns])

# Prepare the kinetic models for Metabolic Control Analysis (MCA)
kmodel_MUT.prepare()
kmodel_MUT.compile_mca(mca_type=NET, sim_type=QSSA, parameter_list=parameter_list, ncpu=ncpu)

# Load the WT steady state samples and use PCA to determine which one is closest to the mean
samples_WT = pd.read_csv(path_to_tfa_samples_WT, index_col=0)

# Clean the samples to only include active fluxes and metabolite concentrations
active_fluxes = []
for r in tmodel_WT.reactions:
    if r.lower_bound > 0:
        active_fluxes.append(r.id)
    else:
        active_fluxes.append(r.reverse_id)

active_vars = [i for i in tmodel_WT.variables.keys() if i.startswith('LC_')]

samples_WT_clean = pd.DataFrame()
for name, vals in samples_WT.iteritems():
    if name in active_fluxes or name in active_vars:
        samples_WT_clean[name] = vals

samples_WT_clean = samples_WT_clean.loc[0:599, :]

# Select the reference sample for WT
reference_sample_index_WT = 66
reference_sample_WT = samples_WT_clean.loc[reference_sample_index_WT,:]
tfa_ix = 17

parameter_population = load_parameter_population(path_to_param_output_MUT.format(tfa_ix))

# Load tfa sample and kinetic parameters into kinetic model
path_to_tfa_samples = path_to_tfa_samples_MUT
tfa_sample = pd.read_csv(path_to_tfa_samples, header=0, index_col=0).iloc[tfa_ix]
parameter_set = parameter_population['0']
kmodel_MUT.parameters = parameter_set

# Load steady-state fluxes and concentrations into the scaffold kmodel
fluxes = load_fluxes(tfa_sample, tmodel_MUT, kmodel_MUT,
                        density=DENSITY,
                        ratio_gdw_gww=GDW_GWW_RATIO,
                        concentration_scaling=CONCENTRATION_SCALING,
                        time_scaling=TIME_SCALING)
concentrations = load_concentrations(tfa_sample, tmodel_MUT, kmodel_MUT,
                                        concentration_scaling=CONCENTRATION_SCALING)

# Fetch equilibrium constants
load_equilibrium_constants(tfa_sample, tmodel_MUT, kmodel_MUT,
                            concentration_scaling=CONCENTRATION_SCALING,
                            in_place=True)

fcc = kmodel_MUT.flux_control_fun(fluxes, concentrations, [parameter_set]).mean('sample')
ccc = kmodel_MUT.concentration_control_fun(fluxes, concentrations, [parameter_set]).mean('sample')

# Find where the fcc is smaller that 1e-5 and make it zero
small_fcc = np.where(fcc.abs() < 1e-5)
for i, u in zip(small_fcc[0], small_fcc[1]):
    fcc.iloc[i, u] = 0

small_ccc = np.where(ccc.abs() < 1e-5)
for i, u in zip(small_ccc[0], small_ccc[1]):
    ccc.iloc[i, u] = 0



"""
NRA setup
"""
# Remove bounds on concentrations in the thermo model - we only constrain fold changes in concentrations
for this_LC in tmodel_MUT.log_concentration:
    this_LC.variable.ub = 100
    this_LC.variable.lb = -100

# from NRAplus.core.model_parallel import NRAmodel_parallel
nmodel = NRAmodel_parallel(tmodel_MUT, tfa_sample, fcc, ccc, type=NET)

reference_concentrations_dict = {c.id: np.exp(tfa_sample.loc[c.name]) for c in nmodel.log_concentration}
reference_concentrations = pd.Series(reference_concentrations_dict)
ll = prepare_metabolites(nmodel, reference_concentrations, ccc, processes=int(ncpu//2))

# from NRAplus.core.model_parallel import NRAmodel_parallel
nmodel = NRAmodel_parallel(tmodel_MUT, tfa_sample, fcc, ccc, type=NET)
reference_fluxes = get_reaction_data(tmodel_MUT, tfa_sample)
ll = prepare_reactions(nmodel, reference_fluxes, fcc, type="NET", processes=int(ncpu//2))

# Folder containing the JSON files
INPUT_FOLDER = './tmp_met_json_MUT'
OUTPUT_FILE = './tmp_met_json_MUT/combined.json'

# Function to load a single JSON file
def load_json(file_path):
    with open(file_path, 'r') as file:
        return orjson.loads(file.read())

# Get all JSON file paths using os.scandir for better performance
json_files = [entry.path for entry in os.scandir(INPUT_FOLDER) if entry.is_file() and entry.name.endswith('.json')]

# Combined structure
combined_data = {
    'variables': [],
    'constraints': []
}

# Use ThreadPoolExecutor with max_workers based on the number of CPU cores
max_workers = min(32, (multiprocessing.cpu_count() or 1) * 5)
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    future_to_file = {executor.submit(load_json, file): file for file in json_files}

    for future in as_completed(future_to_file):
        try:
            data = future.result()
            combined_data['variables'].extend(data.get('variables', []))
            combined_data['constraints'].extend(data.get('constraints', []))
        except Exception as e:
            print(f"Error processing {future_to_file[future]}: {e}")

# Write the combined data to the output JSON file using orjson
with open(OUTPUT_FILE, 'wb') as outfile:
    outfile.write(orjson.dumps(combined_data, option=orjson.OPT_INDENT_2))

print(f"Combined {len(json_files)} JSON files into {OUTPUT_FILE}")


# Folder containing the JSON files
INPUT_FOLDER = './tmp_rxn_json'
OUTPUT_FILE = './tmp_rxn_json/combined.json'

# Function to load a single JSON file
def load_json(file_path):
    with open(file_path, 'r') as file:
        return orjson.loads(file.read())

# Get all JSON file paths using os.scandir for better performance
json_files = [entry.path for entry in os.scandir(INPUT_FOLDER) if entry.is_file() and entry.name.endswith('.json')]

# Combined structure
combined_data = {
    'variables': [],
    'constraints': []
}

# Use ThreadPoolExecutor with max_workers based on the number of CPU cores
max_workers = min(32, (multiprocessing.cpu_count() or 1) * 5)
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    future_to_file = {executor.submit(load_json, file): file for file in json_files}

    for future in as_completed(future_to_file):
        try:
            data = future.result()
            combined_data['variables'].extend(data.get('variables', []))
            combined_data['constraints'].extend(data.get('constraints', []))
        except Exception as e:
            print(f"Error processing {future_to_file[future]}: {e}")

# Write the combined data to the output JSON file using orjson
with open(OUTPUT_FILE, 'wb') as outfile:
    outfile.write(orjson.dumps(combined_data, option=orjson.OPT_INDENT_2))

print(f"Combined {len(json_files)} JSON files into {OUTPUT_FILE}")


# Combine both files to one json file using orjson
OUTPUT_FILE_MET = './tmp_met_json_MUT/combined.json'
OUTPUT_FILE_RXN = './tmp_rxn_json/combined.json'
OUTPUT_FILE = './combined.json'

with open(OUTPUT_FILE_MET, 'r') as file:
    data_met = orjson.loads(file.read())

with open(OUTPUT_FILE_RXN, 'r') as file:
    data_rxn = orjson.loads(file.read())

combined_data = { 'variables': [], 'constraints': [] }
combined_data['variables'].extend(data_met['variables'])
combined_data['constraints'].extend(data_met['constraints'])
combined_data['variables'].extend(data_rxn['variables'])
combined_data['constraints'].extend(data_rxn['constraints'])

with open(OUTPUT_FILE, 'wb') as outfile:
    outfile.write(orjson.dumps(combined_data, option=orjson.OPT_INDENT_2))

print(f"Combined both JSON files into {OUTPUT_FILE}")


# Read the combined JSON file with orjson
OUTPUT_FILE = './combined.json'
with open(OUTPUT_FILE, 'r') as file:
    combined_data = orjson.loads(file.read())

nra_model = add_vars_cons_from_dict(nmodel, combined_data)

nra_model._prepare_thermodynamics(reference_fluxes,type=NET)
# Repair the model dicts
nra_model.regenerate_variables()
nra_model.regenerate_constraints()

save_json_nra_model(nra_model, path_to_nra_model)
print('NRA model saved successfully.')