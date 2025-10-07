"""
Code to create kinetic model based on the pytfa model provided
The produced kinetic model is a draft version and might be unusable if:
i) The reactant - product pairs are not aligned
ii) the hill_coefficients are set to none and not a value like 1.0

After doing manual changes the model is ready for sampling of kinetic parameters
"""

import configparser
import os

from skimpy.analysis.oracle.minimum_fluxes import MinFLux, \
    MinFLuxVariable
from pytfa.io.json import load_json_model
from pytfa.analysis import variability_analysis
from skimpy.io.generate_from_pytfa import FromPyTFA
from skimpy.io.yaml import export_to_yaml
from skimpy.core.compartments import Compartment
from skimpy.utils.general import sanitize_cobra_vars

PHYSIOLOGY = 'WT'

# Read configuration from config.ini
config = configparser.ConfigParser()
config_path = os.path.join(os.path.dirname(__file__), 'config.ini')
config.read(config_path)

# Path to data and model from config, using base_dir
base_dir = config['paths']['base_dir']
path_to_tmodel = os.path.abspath(os.path.join(base_dir, config['paths'][f'path_to_tmodel_{PHYSIOLOGY}']))
path_to_kmodel = os.path.abspath(os.path.join(base_dir, config['paths'][f'path_to_kmodel_{PHYSIOLOGY}']))

# Import and set solver
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

# Convert to kinetic model
sol_fdp = tmodel.optimize()

# Metabolites that are not treated as normal products / reactants
small_molecules = ['h_c', 'h_e', 'h_m', 'h_i', 'h_x', 'h_r', 'h_n', 'h_g']

# Metabolites that will not be part of the kinetics of the model
reactants_to_exclude = ['na1_c', 'na1_e']

# Build the kinetic model
print('Building kinetic model from TFA model...')
# Get concentration scaling from config
scaling_section = 'scaling'
concentration_scaling = float(config[scaling_section].get('CONCENTRATION_SCALING', 1e6))
model_gen = FromPyTFA(small_molecules=small_molecules,
                      reactants_to_exclude=reactants_to_exclude,
                      max_revesible_deltag_0=50)

kmodel = model_gen.import_model(tmodel,
                                sol_fdp.raw,
                                concentration_scaling_factor=concentration_scaling)

# Add and map compartments
print('Adding and mapping compartments...')
for c in tmodel.compartments:
    comp = Compartment(name=c)
    kmodel.add_compartment(comp)

for met in tmodel.metabolites:
    comp = kmodel.compartments[met.compartment]
    kin_met = sanitize_cobra_vars(met.id)
    if kin_met in kmodel.reactants:
        kmodel.reactants[kin_met].compartment = comp
    if kin_met in kmodel.parameters:
        kmodel.parameters[kin_met].compartment = comp

# Add volume parameters
print('Adding reference cell volume parameters...')
reference_cell_volume = {'cell_volume_c': 1766.,  # micrometersˆ3 from BioNumbers
                         'cell_volume_e': 1766.,
                         'cell_volume_m': 1766.,
                         'cell_volume_x': 1766.,
                         'cell_volume_r': 1766.,
                         'cell_volume_n': 1766.,
                         'cell_volume_i': 1766.,
                         'cell_volume_g': 1766.,
                         'cell_volume_l': 1766.,
                         'volume_c': 0.720 * 1766.,  # Luby-Phelps K. 2013
                         'volume_e': 1766.,
                         'volume_m': 0.076 * 1766.,  # m 13% of cell volume from BioNumbers
                         'volume_x': 0.010 * 1766.,
                         'volume_r': 0.054 * 1766.,
                         'volume_n': 0.100 * 1766.,
                         'volume_i': 0.010 * 1766.,
                         'volume_g': 0.020 * 1766.,
                         'volume_l': 0.010 * 1766.,
                         }

kmodel.parameters = reference_cell_volume

print('Setting Hill coefficients to 1.0 where needed...')
for this_param in kmodel.parameters.keys():
    if this_param.startswith('hill_coefficient'):
        kmodel.parameters[this_param].value = 1.0  # otherwise none which will give wrong kinetic sampling results

kmodel.prepare()

# Export the kinetic model
# print('Exporting kinetic model to:', path_to_kmodel)
# export_to_yaml(kmodel, path_to_kmodel)
