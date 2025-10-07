import numpy as np
import pandas as pd
from skimpy.analysis.oracle import *
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append("../../../../NRAplus/NRAplus")
from NRAplus.core.constraints import ObjectiveConstraint

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
path_to_enzyme_variability_analysis = os.path.join(base_dir, config['paths']['path_to_enzyme_variability_analysis'])
path_to_essential_enzymes = os.path.join(base_dir, config['paths']['path_to_essential_enzymes'])

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



# NRA setup options
max_enz_mod = 250
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

prev_sol = pd.read_csv(path_to_solutions.format(nra_model.max_enzyme_modifications), index_col=0)

var_ids = list(range(len(nra_model.solver.variables)))
var_vals = [i[0] for i in prev_sol.values]
var_vals = [prev_sol.loc[i.name][0] for i in nra_model.solver.variables]
import cplex

nra_model.solver.problem.MIP_starts.add(cplex.SparsePair(ind=var_ids, val=var_vals), nra_model.solver.problem.MIP_starts.effort_level.repair, 'PreviousSolution')

nra_model.solver.configuration.presolve = True

# Solve the optimization problem
sol = nra_model.optimize()
print('--------------------------------')
print('Enzymes changed {}'.format(nra_model.max_enzyme_modifications))
print("The value of the objective function is: {} and solver status is {}".format(sol.objective_value, sol.status))


# Force the objective function to be equal to the value I get when max enz mod is 200
# Create a copy of the NRA model and introduce a constraint that fixes the objective value

# Get the objective value
sol_original = pd.read_csv(path_to_solutions.format(max_enz_mod), index_col=0)

# Get the values of the auxilariy variables
aux_vars = [v for v in sol_original.index if v.startswith('AUXV_')]
# Get the value of the objective function by adding the values of the auxiliary variables
obj_value = sol_original.loc[aux_vars].sum().values[0]
threshold = 0.01
expr = nra_model.objective.expression
lb = 49.606
ub = 46.85985328185052*1.1 # 10% of the best solution
nra_model.add_constraint(ObjectiveConstraint,
                            hook=nra_model,
                            expr=expr,
                            id_='min_obj_value',
                            lb= lb,
                            ub=ub )

nra_model.max_enzyme_modifications = 250
nra_model.constraints['MIC_ALL'].lb =  -len(nra_model.enzyme_up_regulation_variable) + nra_model.max_enzyme_modifications

# Make the objective function value more strict
nra_model.constraints['OC_min_obj_value'].ub = 46.85985328185052*1.1


# Apply tight constraints everywhere
BIGM = 1000
enzyme_use = [i.name for i in nra_model.enzyme_up_down_use_variable]
for enz in enzyme_use:
    enz = enz.split('_',1)[1]
    con = nra_model.constraints['IC_' + enz]
    # Transform 0<= EUU + EDU + BIGM * EUDU <= BIGM
    # to 1<= EUU + EDU + EUDU <= 1
    # This forces either regulation or nothing at all
    con.set_linear_coefficients({nra_model.variables['EUDUSE_' + enz]: 1})
    con.lb = 1
    con.ub = 1

# If EUDUSE is 0 the EU or ED must have a basal value
for enz in enzyme_use:
    enz = enz.split('_',1)[1]
    con = nra_model.constraints['EUC_' + enz]
    con.lb = -BIGM+1e-6

for enz in enzyme_use:
    enz = enz.split('_',1)[1]
    con = nra_model.constraints['EDC_' + enz]
    con.lb = -BIGM+1e-6

enzyme_use_vars = [i.name for i in nra_model.enzyme_up_down_use_variable]
enzyme_use_df = sol_original.loc[enzyme_use_vars]

# Find which variables are active EUDUSE = 0
always_active_df = enzyme_use_df[np.isclose(enzyme_use_df['EUDUSE'], 0)]

enzyme_fold_changes = pd.Series(dtype=float)
for enz in always_active_df.index:
    up_value = sol_original.loc[enz.replace('EUDUSE_', 'EU_')].values[0] 
    down_value = sol_original.loc[enz.replace('EUDUSE_', 'ED_')].values[0]
    enzyme_fold_changes[enz.replace('EUDUSE_', 'EU_')] = up_value
    enzyme_fold_changes[enz.replace('EUDUSE_', 'ED_')] = down_value


# Run the variability analysis
print('Running variability analysis...  ')
from optlang.symbolics import Zero
nra_model.objective = Zero

nra_model.max_fold_enzyme_change = np.log(50)

nra_model.solver.configuration.timeout = 3600

# Load solver configuration from config.ini [solver_for_nra_optimization]
solver_config = config['solver_for_nra_optimization']
nra_model.solver.configuration.tolerances.feasibility = float(solver_config['feasibility_tolerance'])
nra_model.solver.configuration.tolerances.optimality = float(solver_config['optimality_tolerance'])
nra_model.solver.configuration.tolerances.integrality = float(solver_config['integrality_tolerance'])
nra_model.solver.problem.parameters.read.scale = int(solver_config['read_scale'])
nra_model.solver.problem.parameters.emphasis.mip.set(int(solver_config.get('emphasis_mip', '3')))


nra_model.solver.problem.parameters.threads.set(ncpu)
nra_model.solver.problem.parameters.mip.display.set(3)

nra_model.solver.problem.set_results_stream('./cplex_logs_small_model_significant_transporters/tmp_variability_results.txt')

prev_sol = pd.read_csv(path_to_solutions.format(nra_model.max_enzyme_modifications), index_col=0)

var_ids = list(range(len(nra_model.solver.variables)))
var_vals = [i[0] for i in prev_sol.values]
import cplex

nra_model.solver.problem.MIP_starts.add(cplex.SparsePair(ind=var_ids, val=var_vals), nra_model.solver.problem.MIP_starts.effort_level.repair, 'PreviousSolution')

nra_model.solver.configuration.presolve = True

df_var = pd.DataFrame(columns=['min', 'max'])

for var_id in tqdm(enzyme_fold_changes.index):
    var = nra_model.variables[var_id]
    # For min value
    nra_model.objective = nra_model.variables[var_id]
    nra_model.objective_direction = 'min'
    ll = nra_model.optimize()
    df_var.loc[var_id] = [ll.objective_value, 100]
    
    # For max value
    previous_value = enzyme_fold_changes.loc[var_id]    
    nra_model.objective_direction = 'max'
    ll = nra_model.optimize()
    df_var.loc[var_id, 'max'] = ll.objective_value

    print('Variable {} min: {} max: {}'.format(var_id, df_var.loc[var_id]['min'], df_var.loc[var_id]['max']))

    df_var.to_csv(path_to_enzyme_variability_analysis)

print('Variability analysis completed. Results saved to:', path_to_enzyme_variability_analysis)

# Find the essential enzymes
essential_enzymes = []
for enz in always_active_df.index:
    enz = enz.split('EUDUSE_')[1]
    up_var = nra_model.variables['EU_' + enz]
    down_var = nra_model.variables['ED_' + enz]

    up_bounds = df_var.loc['EU_' + enz]
    down_bounds = df_var.loc['ED_' + enz]

    if np.isclose(up_bounds['min'], 0) and np.isclose(down_bounds['min'], 0):
        pass
    else:
        essential_enzymes.append(enz)

# Save the essential enzymes to a file
with open(path_to_essential_enzymes, 'w') as f:
    for enz in essential_enzymes:
        f.write(enz + '\n')
