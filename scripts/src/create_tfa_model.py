"""
create_tfa_model.py
-------------------
This script builds a thermodynamically curated TFA model from a MATLAB model, imposes constraints, and performs variability analysis and sampling.

Description:
Based on the provided .mat file, the code extracts the FDP and the bounds of all the variables (LC, DG, ...).
It then imposes basal fluxes, minimum thermodynamic displacement, and forces diffusion transport reactions to operate close to equilibrium.
Lastly, the code does variability analysis and runs an initial sampling of steady-state sample points.
"""

import configparser
import os

import numpy as np
import pandas as pd
from scipy.io import loadmat
import pytfa
from pytfa.io import import_matlab_model, load_thermoDB
from pytfa.analysis import variability_analysis, apply_reaction_variability, apply_generic_variability, sample
from pytfa.optim.variables import DeltaG, DeltaGstd, ThermoDisplacement, LogConcentration
from pytfa.io.json import save_json_model
from pytfa.optim.relaxation import relax_dgo
from pytfa.redgem.utils import remove_blocked_reactions
from skimpy.analysis.oracle import *
from scipy.sparse import *

PHYSIOLOGY = 'WT'


# Read configuration from config.ini
print('Reading configuration from config.ini...')
config = configparser.ConfigParser()
config_path = os.path.join(os.path.dirname(__file__), 'config.ini')
config.read(config_path)

# Path to data and model from config, using base_dir
print('Setting up paths from config...')
base_dir = config['paths']['base_dir']
path_to_mat_tmodel = os.path.abspath(os.path.join(base_dir, config['paths'][f'path_to_mat_tmodel_{PHYSIOLOGY}']))
path_to_thermodb = os.path.abspath(os.path.join(base_dir, config['paths']['path_to_thermodb']))
path_to_tmodel = os.path.abspath(os.path.join(base_dir, config['paths'][f'path_to_tmodel_{PHYSIOLOGY}']))

# Load cobra model from mat file
print('Importing MATLAB model...')
cmodel = import_matlab_model(path_to_mat_tmodel)

# Convert to a thermodynamics model
print('Loading thermodynamics database and building TFA model...')
thermo_data = load_thermoDB(path_to_thermodb)
tmodel = pytfa.ThermoModel(thermo_data, cmodel)


# Setup solver from config
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

# TFA conversion
print('Preparing and converting model to TFA...')
tmodel.prepare()
tmodel.convert(add_displacement=True)
print('Optimizing initial TFA model...')
solution = tmodel.optimize()

# Import variable bounds from mat file
print('Loading variable bounds from MATLAB file...')
mat_data = loadmat(path_to_mat_tmodel)
mat_model = mat_data['model_final'][0, 0]
varname_index = {e[0][0]: i for i, e in enumerate(mat_model['varNames'])}

# Extract DGo values from the mat file
print('Setting DGo variable bounds...')
for dgo in tmodel.delta_gstd:
    var_name = dgo.name
    i = varname_index[var_name]

    lb = mat_model['var_lb'][i][0]
    ub = mat_model['var_ub'][i][0]
    if dgo.variable.ub < lb:
        dgo.variable.ub = ub
        dgo.variable.lb = lb
    else:
        dgo.variable.lb = lb
        dgo.variable.ub = ub

# Extract LogConcentration values from the mat file
print('Setting LogConcentration variable bounds...')
for lc in tmodel.log_concentration:
    var_name = lc.name
    if var_name in varname_index:
        i = varname_index[var_name]
    else:
        continue

    lb = mat_model['var_lb'][i][0]
    ub = mat_model['var_ub'][i][0]
    if lc.variable.ub < lb:
        lc.variable.ub = ub
        lc.variable.lb = lb
    else:
        lc.variable.lb = lb
        lc.variable.ub = ub

# Extract net-flux values from the mat file
print('Setting net-flux variable bounds...')
for rxn in tmodel.reactions:
    forward_variable = rxn.forward_variable
    reverse_variable = rxn.reverse_variable

    i = varname_index['NF_{}'.format(rxn.id)]

    lb = mat_model['var_lb'][i][0]
    ub = mat_model['var_ub'][i][0]
    rxn.lower_bound = lb
    try:
        rxn.upper_bound = ub
    except:
        rxn.upper_bound = lb

# Make sure the bounds for biomass flux is correctly set
print('Setting bounds for biomass flux...')
i = varname_index['NF_{}'.format('biomass')]
lb = mat_model['var_lb'][i][0]
ub = mat_model['var_ub'][i][0]
print('Biomass lower bound:', lb)
print('Biomass upper bound:', ub)

tmodel.reactions.biomass.bounds = (0.0231, 0.0347)
print('Optimizing after setting biomass bounds...')
tmodel.optimize()

# Remove blocked reactions
print('Removing blocked reactions...')
rxns_rem = remove_blocked_reactions(tmodel)

rem_mets = []
for i in tmodel.metabolites:
    if len(i.reactions) == 0:
        rem_mets.append(i)

print('Removing metabolites with no reactions...')
tmodel.remove_metabolites(rem_mets)

# Impose basal flux constraints
print('Imposing basal flux constraints...')
MIN_FLUX = 1e-6
add_min_flux_requirements(tmodel, MIN_FLUX, inplace=True)
res = relax_min_flux(tmodel)
tmodel = res[0].copy()

print('Setting ATP, ADP, AMP log concentration bounds...')
ATP = 4.67e-3
ADP = 5.69e-4
AMP = 4.23e-5
slack = 0.4
tmodel.variables['LC_atp_c'].lb = np.log(ATP*(1-slack))
tmodel.variables['LC_atp_c'].ub = np.log(ATP*(1+slack))

tmodel.variables['LC_adp_c'].lb = np.log(ADP*(1-slack))
tmodel.variables['LC_adp_c'].ub = np.log(ADP*(1+slack))

tmodel.variables['LC_amp_c'].lb = np.log(AMP*(1-slack))
tmodel.variables['LC_amp_c'].ub = np.log(AMP*(1+slack))

tmodel.reactions.biomass.bounds = (0.0231, 0.0347)

print('Optimizing after setting ATP/ADP/AMP bounds...')
try:
    sol = tmodel.optimize()
except:
    print('Relaxing DGo values for ATP/ADP/AMP...')
    res = relax_dgo(tmodel)
    tmodel = res[0].copy()

solution = tmodel.optimize()
# tmodel.reactions.biomass.lower_bound = 0.8 * solution.objective_value

# Add all missing DGs
print('Adding missing DGs (first pass)...')
tmodel = add_undefined_delta_g(tmodel,
                               solution,
                               delta_g_std=0.0,
                               delta_g_std_err=10.0,
                               add_displacement=True,
                               inplace=True,
                               exclude_reactions= [tmodel.reactions.biomass.id])  # NOTE: exchange reactions do not have thermodynamic properties

# Add all missing DGs
print('Adding missing DGs (second pass)...')
tmodel = add_undefined_delta_g(tmodel,
                               solution,
                               delta_g_std=0.0,
                               delta_g_std_err=100.0,
                               add_displacement=True,
                               inplace=True)  # NOTE: exchange reactions do not have thermodynamic properties

tmodel.variables['DGo_biomass'].lb = -100
tmodel.variables['DGo_biomass'].ub = +100
tmodel.variables['DG_biomass'].lb = -1000
tmodel.variables['DG_biomass'].ub = +1000

# Force a minimal thermodynamic displacement
print('Copying model for FDP and running variability analysis...')
tmodel_fdp = tmodel.copy()
tva_fluxes = variability_analysis(tmodel_fdp, kind='reactions')

min_log_displacement = 1e-3
print('Adding minimum log displacement...')
tmodel_fdp = add_min_log_displacement(tmodel_fdp,
                                      min_log_displacement,
                                      tva_fluxes=tva_fluxes,
                                      inplace=False)

print('Optimizing after adding minimum log displacement...')
try:
    sol = tmodel_fdp.optimize()
except:
    print('Relaxing DGo values for FDP...')
    res = relax_dgo(tmodel_fdp)
    tmodel_fdp = res[0].copy()

from pytfa.analysis.variability import _variability_analysis_element
def element_variability_analysis(tmodel, varname, kind=None, verbal=0):
    orig_objective = tmodel.objective
    if kind == 'reactions':
        if verbal: print('MinMax for reaction ' + varname)
        rxn = tmodel.reactions.get_by_id(varname)
        tva_min = _variability_analysis_element(tmodel, rxn.forward_variable - rxn.reverse_variable, 'min')
        tva_max = _variability_analysis_element(tmodel, rxn.forward_variable - rxn.reverse_variable, 'max')
        tmodel.objective = orig_objective  # change to the model's original objective function
        return [tva_min, tva_max]
    else:
        if verbal: print('MinMax for variable ' + varname)
        tva_min = _variability_analysis_element(tmodel, tmodel.variables[varname], 'min')
        tva_max = _variability_analysis_element(tmodel, tmodel.variables[varname], 'max')
        tmodel.objective = orig_objective  # change to the model's original objective function
        return [tva_min, tva_max]

# Find the reactions that are diffusion transporters
print('Identifying diffusion transport reactions...')
transport_reactions = []
for i in range(0, len(mat_model['rxns'])):
    if mat_model['isTrans'][i] == 1:
        try:
            if len(tmodel_fdp.reactions.get_by_id(mat_model['rxns'][i][0][0]).metabolites) == 2:  # only diffusion
                transport_reactions.append(tmodel.reactions.get_by_id(mat_model['rxns'][i][0][0]).id)
        except:  # not needed
            continue

print('Running variability analysis for transport reactions...')
thermo_disp_varnames = [tmodel_fdp.thermo_displacement.get_by_id(r_id).name for r_id in transport_reactions]
tva_disp = [element_variability_analysis(tmodel_fdp, i) for i in thermo_disp_varnames]
tva_disp = pd.DataFrame(tva_disp, index=thermo_disp_varnames)

MAX_LOG_TRANSPORT_DISPLACEMENT = np.log(0.9)
for r_id in transport_reactions:
    variable = tmodel_fdp.thermo_displacement.get_by_id(r_id).variable
    min_disp, max_disp = tva_disp.loc[variable.name, :]

    ub_old = variable.ub
    lb_old = variable.lb

    print('Transport reaction {}: max_disp={}, min_disp={}'.format(r_id, max_disp, min_disp))

    if -MAX_LOG_TRANSPORT_DISPLACEMENT > min_disp > 0:
        print("UB: {}, NEW {} MIN {} MAX {}".format(variable.ub, -MAX_LOG_TRANSPORT_DISPLACEMENT, min_disp, max_disp))
        variable.ub = -MAX_LOG_TRANSPORT_DISPLACEMENT

    elif MAX_LOG_TRANSPORT_DISPLACEMENT < max_disp < 0:
        print("UB: {}, NEW {} MIN {} MAX {}".format(variable.lb, MAX_LOG_TRANSPORT_DISPLACEMENT, min_disp, max_disp))
        variable.lb = MAX_LOG_TRANSPORT_DISPLACEMENT
    else:
        print('Reaction needs to be displaced from eqiulibrium {}'.format(r_id))
        continue
    try:
        tmodel_fdp.optimize()
    except:
        print('Reaction needs to be displaced from eqiulibrium {}'.format(r_id))
        variable.ub = ub_old
        variable.lb = lb_old

print('Copying model after imposing thermodynamic constraints...')
tmodel_after_thermo = tmodel_fdp.copy()
# Do variability analysis and impose the new bounds
print('Running final variability analysis and tightening bounds...')
tva_fluxes = variability_analysis(tmodel_fdp, kind='reactions')
thermo_vars = [DeltaG, DeltaGstd, ThermoDisplacement, LogConcentration]
tva_thermo = variability_analysis(tmodel_fdp, kind=thermo_vars)
tight_model = apply_reaction_variability(tmodel_fdp, tva_fluxes)
tight_model = apply_generic_variability(tight_model, tva_thermo)
tight_model.objective = tight_model.reactions.biomass
print('Optimizing final tight model...')
sol = tight_model.optimize()

# Save curated version of the model
# save_json_model(tight_model, path_to_tmodel)