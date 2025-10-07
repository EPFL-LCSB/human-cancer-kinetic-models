import pickle
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm
import configparser
import os
from skimpy.analysis.oracle.load_pytfa_solution import load_fluxes, \
    load_concentrations, load_equilibrium_constants
from skimpy.core.parameters import ParameterValuePopulation, \
    load_parameter_population
from skimpy.utils.tensor import Tensor


# Read scaling values from the config file
config = configparser.ConfigParser()
config_path = '../src/config.ini'
config.read(config_path)
scaling_section = 'scaling'
CONCENTRATION_SCALING = float(config[scaling_section].get('CONCENTRATION_SCALING', 1e6))
TIME_SCALING = float(config[scaling_section].get('TIME_SCALING', 1))
DENSITY = float(config[scaling_section].get('DENSITY', 1200))
GDW_GWW_RATIO = float(config[scaling_section].get('GDW_GWW_RATIO', 0.25))
flux_scaling_factor = 1e-3 * (GDW_GWW_RATIO * DENSITY) * CONCENTRATION_SCALING / TIME_SCALING

def get_conc_control_coefficients(li,ui,ss, samples, tmodel_, kmodel_, path_to_lambda_values, path_to_param_output, path_to_ccc = '../CCCs/conc_control_coeff_ss_{}_{}_{}.pkl'):    
    # li ui and ss are the lower index, upper index and the steady state number
    # li, ui, ss = args
    
    # Check which models passed the pruning criteria
    lambda_df = pd.read_csv(path_to_lambda_values.format(ss), index_col=0)
    eig_cutoff = -3/(np.log(2)/samples.biomass.iloc[ss])
    stable_fast_models = np.where(lambda_df.loc[:,'max_eig'] < eig_cutoff)[0]

    # Load the parameter population
    parameter_population = load_parameter_population(path_to_param_output.format(ss), lower_index=li, upper_index=ui)
    
    # Do MCA only for the stable and fast models
    stable_fast_index = [i for i in parameter_population._index.keys() if int(i) in stable_fast_models]
    samples_to_simulate = stable_fast_index

    conc_control_data = []

    for ix in tqdm(samples_to_simulate):
        # Add the kinetic parameters of the current set to the kinetic model
        kmodel_.parameters = parameter_population[ix]
        tfa_sample = samples.loc[ss]

        # Load steady state fluxes and concentrations from the TFA sample into the kinetic model
        fluxes = load_fluxes(tfa_sample, tmodel_, kmodel_,
                            density=DENSITY,
                            ratio_gdw_gww=GDW_GWW_RATIO,
                            concentration_scaling=CONCENTRATION_SCALING,
                            time_scaling=TIME_SCALING)

        concentrations = load_concentrations(tfa_sample, tmodel_, kmodel_,
                                            concentration_scaling=CONCENTRATION_SCALING)

        conc_control_coeff = kmodel_.concentration_control_fun(fluxes, concentrations, [parameter_population[ix]])
        conc_control_data.append(conc_control_coeff._data)

    if len(conc_control_data) == 0:
        return
    # Make tensor for the population level
    ccc_data = np.concatenate(conc_control_data, axis=2)

    conc_index = pd.Index(kmodel_.independent_variables_names, name="concentration")
    parameter_index = pd.Index(kmodel_.concentration_control_fun.parameter_elasticity_function.respective_variables, name="parameter")
    sample_index = pd.Index(samples_to_simulate, name="sample")

    # Get the mean concentration control coefficients for this sample
    conc_control_coeff = Tensor(ccc_data, [conc_index, parameter_index, sample_index])
    # Saving
    with open(path_to_ccc.format(ss,li,ui-1), 'wb') as f:
        pickle.dump(conc_control_coeff, f)

    del conc_control_data

    del conc_control_coeff


def get_cccs_for_metabolite(ss, metabolites, path='../CCCs/conc_control_coeff_ss_{}_{}_{}.pkl'):
    '''Get the concentration control coefficients for a specific steady state and multiple metabolites'''

    # Load the CCC data for all metabolites at once
    ccc_met_slice_dict = {metabolite: pd.DataFrame() for metabolite in metabolites}

    for i in range(0, 100, 10):
        try:
            # Open the file once and load the CCC data
            with open(path.format(ss, i, i + 9), 'rb') as f:
                ccc = pickle.load(f)
            
            # For each metabolite, extract the corresponding slice and add it to the dictionary
            for met in metabolites:
                ccc_met_slice_tmp = ccc.slice_by('concentration', met)
                ccc_met_slice_dict[met] = pd.concat([ccc_met_slice_dict[met], ccc_met_slice_tmp], axis=1)
        except:
            pass

    # Update column names for each metabolite
    for metabolite in metabolites:
        ccc_met_slice_dict[metabolite].columns = ['{},'.format(ss) + str(col) for col in ccc_met_slice_dict[metabolite].columns]

    return ccc_met_slice_dict


def get_cccs_parallel(ss_range, metabolites=['atp_c'], path='../CCCs/conc_control_coeff_ss_{}_{}_{}.pkl', n_cores=1):
    """
    Parallelize the CCCs extraction over steady states and optionally handle multiple metabolites.

    Parameters:
        ss_range (iterable): Range or list of steady states to process.
        metabolites (list or str): Metabolite name(s). Can be a single string or a list of strings.
        path (str): Path template for CCC files.
        n_cores (int, optional): Number of cores for parallelization. Defaults to all available cores.

    Returns:
        dict: A dictionary with metabolite names as keys and their corresponding DataFrames as values.
    """
    # Ensure metabolites is a list for consistent processing
    if isinstance(metabolites, str):
        metabolites = [metabolites]

    results_dict = {}

    # Use ThreadPoolExecutor to process steady states in parallel
    with ThreadPoolExecutor(max_workers=n_cores) as executor:
        results = list(tqdm(
            executor.map(get_cccs_for_metabolite, ss_range, [metabolites] * len(ss_range), [path] * len(ss_range)),
            total=len(ss_range),
            desc="Processing Steady States"
        ))

    # Combine all non-None results into a single dictionary
    for result in results:
        for metabolite, df in result.items():
            if metabolite not in results_dict:
                results_dict[metabolite] = df
            else:
                results_dict[metabolite] = pd.concat([results_dict[metabolite], df], axis=1)

    return results_dict
