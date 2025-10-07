import numpy as np
import pandas as pd
from skimpy.analysis.oracle import *
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append("../../../../NRAplus/NRAplus") # Adds higher directory to python modules path.
sys.path.append('../')

from utils.nra_save_custom_json import load_json_nra_model
from pytfa.io.json import load_json_model


from pytfa.optim.variables import ReactionVariable
from pytfa.optim.constraints import ReactionConstraint



# Load paths from config.ini
import configparser
import os

config = configparser.ConfigParser()
config.read(os.path.abspath(os.path.join(os.path.dirname(__file__), 'config.ini')))

ncpu = int(config['global']['ncpu'])

base_dir = config['paths']['base_dir']
path_to_tfa_samples_MUT = os.path.join(base_dir, config['paths']['path_to_samples_MUT'])
path_to_tfa_samples_WT = os.path.join(base_dir, config['paths']['path_to_samples_WT'])
path_to_tmodel_MUT = os.path.join(base_dir, config['paths']['path_to_tmodel_MUT'])
path_to_tmodel_WT = os.path.join(base_dir, config['paths']['path_to_tmodel_WT'])
path_to_nra_model = os.path.join(base_dir, config['paths']['path_to_nra_model'])
path_to_solutions = os.path.join(base_dir, config['paths']['path_to_solutions'])


tmodel_MUT = load_json_model(path_to_tmodel_MUT)
tmodel_WT = load_json_model(path_to_tmodel_WT)

# Find how many reactions have opposite directionality in the two models.
# We will not include them in the objective function.
diff_dir_rxns = []
mut_rxns_ids = [rxn.id for rxn  in tmodel_MUT.reactions]
for rxn in tmodel_WT.reactions:
    if rxn.id in mut_rxns_ids:
        rxn_mut = tmodel_MUT.reactions.get_by_id(rxn.id)
        if rxn_mut.lower_bound*rxn.lower_bound < 0:
            diff_dir_rxns.append(rxn.id)

print('Number of reactions with opposite directionality in the two models:', len(diff_dir_rxns))

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

# Use PCA to determine which one is closest to the mean
samples_WT_clean = samples_WT_clean.loc[0:599, :]

# Select the reference sample for WT
reference_sample_index = 66
reference_sample_WT = samples_WT_clean.loc[reference_sample_index,:]

# Load the MUT steady state samples
tfa_ix = 17
tfa_sample = pd.read_csv(path_to_tfa_samples_MUT, header=0, index_col=0).iloc[tfa_ix]

print('Loading NRA model...')
nra_model = load_json_nra_model(path_to_nra_model)

# Some tmp classes
class AuxiliaryVariable(ReactionVariable):
    """
    Class to represent the flux deviation z = abs(ln(v_WT)-ln(v_MUT))
    """
    def __init__(self,reaction,**kwargs):
        ReactionVariable.__init__(self, reaction, **kwargs)
    prefix = 'AUXV_'

class AuxiliaryCouplingPos(ReactionConstraint):
    """
    Class to represent the z >= ln(v_WT)-ln(v_MUT)
    """
    def __init__(self, reaction, expr, **kwargs):
        ReactionConstraint.__init__(self, reaction, expr, **kwargs)
    prefix = 'AUXCP_'

class AuxiliaryCouplingNeg(ReactionConstraint):
    """
    Class to represent the z >= ln(v_MUT)-ln(v_WT)
    """
    def __init__(self, reaction, expr, **kwargs):
        ReactionConstraint.__init__(self, reaction, expr, **kwargs)
    prefix = 'AUXCN_'


# Choose as objective function only reactions that belong to the core subsystems
core_ss = ['Glycolysis/gluconeogenesis', 'Citric acid cycle', 'Pentose phosphate pathway', 'Glutamate metabolism',
'ROS detoxification', 'Glycine, serine, alanine, and threonine metabolism', 'Urea cycle', 'Arginine and proline metabolism',
'Purine synthesis', 'Pyrimidine synthesis', 'ETC_Rxns', 'NAD metabolism']

log_wt_fluxes = np.log(reference_sample_WT[active_fluxes])

original_value = 0
for rxn in nra_model.reactions:
    # Check if the reaction belongs to the core subsystems
    if rxn not in nra_model.boundary and rxn.subsystem in core_ss and rxn.id not in diff_dir_rxns:
        try:
            wt_flux = log_wt_fluxes[rxn.id]
        except:
            wt_flux = log_wt_fluxes[rxn.reverse_id]
            
        # Create the auxiliary variable (z>=0)
        aux_var = nra_model.add_variable(kind=AuxiliaryVariable, hook=rxn, lb=0, ub=100)

        # Add the coupling constraints
        expr_pos = aux_var + nra_model.log_flux.get_by_id(rxn.id)
        nra_model.add_constraint(kind=AuxiliaryCouplingPos, hook=rxn, expr=expr_pos, lb=wt_flux)

        expr_neg = aux_var - nra_model.log_flux.get_by_id(rxn.id)
        nra_model.add_constraint(kind=AuxiliaryCouplingNeg, hook=rxn, expr=expr_neg, lb=-wt_flux)

        mut_flux = np.log(np.abs(tfa_sample[rxn.id]-tfa_sample[rxn.reverse_id]))
        original_value += abs(wt_flux - mut_flux)

print("The original value of the objective function is: {}".format(original_value))


# The objective is the minimization of the sum of the auxiliary variables
obj_expr = 0
for var in nra_model.variables:
    if var.name.startswith('AUXV_'):
        obj_expr += var

nra_model.objective = obj_expr
nra_model.objective_direction = 'min'


# Allow for knockouts
for var in nra_model.enzyme_down_regulation_variable:
    var.variable.ub = 100

# Don't allow enzymatic changes to the ATPM reaction
nra_model.variables.EU_ATPM.ub = 1e-3
nra_model.variables.ED_ATPM.ub = 1e-3

# Don't allow enzymatic changes to the biomass reaction
nra_model.variables.EU_biomass.ub = 1e-3
nra_model.variables.ED_biomass.ub = 1e-3

# Make sure the LogFluxes can have values in the range of ln(1e-6) to ln(100)
for var in nra_model.log_flux:
    var.variable.lb = -13.8155 # we avoid np.log(1e-6) because of numerical issues
    var.variable.ub = 4.6052 # we avoid np.log(100) because of numerical issues


for max_enz_mod in [1, 10, 50, 100, 250, 500, 900]:
    # NRA setup options
    nra_model.max_enzyme_modifications = max_enz_mod
    nra_model.max_fold_enzyme_change = np.log(50) # Based on the maximum observed gene expression ratio from the data


    # Load solver configuration from config.ini [solver_for_nra_optimization]
    solver_config = config['solver_for_nra_optimization']
    nra_model.solver.configuration.timeout = int(solver_config.get('timeout', '18000'))
    nra_model.solver.configuration.tolerances.feasibility = float(solver_config['feasibility_tolerance'])
    nra_model.solver.configuration.tolerances.optimality = float(solver_config['optimality_tolerance'])
    nra_model.solver.configuration.tolerances.integrality = float(solver_config['integrality_tolerance'])
    nra_model.solver.problem.parameters.read.scale = int(solver_config['read_scale'])
    nra_model.solver.problem.parameters.emphasis.numerical = int(solver_config['numerical_emphasis'])
    nra_model.solver.problem.parameters.emphasis.mip.set(int(solver_config.get('emphasis_mip', '3')))
    nra_model.solver.problem.parameters.mip.strategy.nodeselect.set(int(solver_config.get('mip_strategy_nodeselect', '1')))
    nra_model.solver.problem.parameters.mip.cuts.flowcovers.set(int(solver_config.get('mip_cuts_flowcovers', '1')))


    nra_model.solver.problem.parameters.threads.set(ncpu)
    nra_model.solver.problem.parameters.mip.display.set(3)
    results_stream = solver_config['results_stream'].format(nra_model.max_enzyme_modifications)
    nra_model.solver.problem.set_results_stream(results_stream)  # Results stream
    
    nra_model.solver.configuration.presolve = True  

    # Solve the optimization problem
    sol = nra_model.optimize()
    print('--------------------------------')
    print('Enzymes changed {}'.format(nra_model.max_enzyme_modifications))
    print("The value of the objective function is: {} and solver status is {}".format(sol.objective_value, sol.status))

    sol.raw.to_csv(path_to_solutions.format(nra_model.max_enzyme_modifications))