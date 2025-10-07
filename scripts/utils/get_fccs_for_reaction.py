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
config_path = os.path.join(os.path.dirname(__file__), 'config.ini')
config.read(config_path)
scaling_section = 'scaling'
CONCENTRATION_SCALING = float(config[scaling_section].get('CONCENTRATION_SCALING', 1e6))
TIME_SCALING = float(config[scaling_section].get('TIME_SCALING', 1))
DENSITY = float(config[scaling_section].get('DENSITY', 1200))
GDW_GWW_RATIO = float(config[scaling_section].get('GDW_GWW_RATIO', 0.25))
flux_scaling_factor = 1e-3 * (GDW_GWW_RATIO * DENSITY) * CONCENTRATION_SCALING / TIME_SCALING

def get_flux_control_coefficients(li,ui,ss, samples, tmodel_, kmodel_, path_to_lambda_values, path_to_param_output, path_to_fcc = '../FCCs/flux_control_coeff_ss_{}_{}_{}.pkl'):    
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

    flux_control_data = []

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

        flux_control_coeff = kmodel_.flux_control_fun(fluxes, concentrations, [parameter_population[ix]])
        flux_control_data.append(flux_control_coeff._data)

    if len(flux_control_data) == 0:
        return
    # Make tensor for the population level
    fcc_data = np.concatenate(flux_control_data, axis=2)

    flux_index = pd.Index(kmodel_.reactions.keys(), name="flux")
    parameter_index = pd.Index(kmodel_.flux_control_fun.parameter_elasticity_function.respective_variables, name="parameter")
    sample_index = pd.Index(samples_to_simulate, name="sample")

    # Get the mean flux control coefficients for this sample
    flux_control_coeff = Tensor(fcc_data, [flux_index, parameter_index, sample_index])
    # Saving
    with open(path_to_fcc.format(ss,li,ui-1), 'wb') as f:
        pickle.dump(flux_control_coeff, f)

    del flux_control_data

    del flux_control_coeff


def get_fccs_for_reaction(ss, reactions, path='../FCCs/flux_control_coeff_ss_{}_{}_{}.pkl'):
    '''Get the flux control coefficients for a specific steady state and multiple reactions'''
    
    # Load the FCC data for all reactions at once
    fcc_rxn_slice_dict = {reaction: pd.DataFrame() for reaction in reactions}
    
    for i in range(0, 100, 10):
        try:
            # Open the file once and load the FCC data
            with open(path.format(ss, i, i + 9), 'rb') as f:
                fcc = pickle.load(f)
            
            # For each reaction, extract the corresponding slice and add it to the dictionary
            for reaction in reactions:
                fcc_rxn_slice_tmp = fcc.slice_by('flux', reaction)
                fcc_rxn_slice_dict[reaction] = pd.concat([fcc_rxn_slice_dict[reaction], fcc_rxn_slice_tmp], axis=1)
        except:
            pass
    
    # Update column names for each reaction
    for reaction in reactions:
        fcc_rxn_slice_dict[reaction].columns = ['{},'.format(ss) + str(col) for col in fcc_rxn_slice_dict[reaction].columns]

    return fcc_rxn_slice_dict


def get_fccs_parallel(ss_range, reactions=['biomass'], path='../FCCs/flux_control_coeff_ss_{}_{}_{}.pkl', n_cores=1):
    """
    Parallelize the FCCs extraction over steady states and optionally handle multiple reactions.

    Parameters:
        ss_range (iterable): Range or list of steady states to process.
        reactions (list or str): Reaction name(s). Can be a single string or a list of strings.
        path (str): Path template for FCC files.
        n_cores (int, optional): Number of cores for parallelization. Defaults to all available cores.

    Returns:
        dict: A dictionary with reaction names as keys and their corresponding DataFrames as values.
    """
    # Ensure reactions is a list for consistent processing
    if isinstance(reactions, str):
        reactions = [reactions]

    results_dict = {}

    # Use ThreadPoolExecutor to process steady states in parallel
    with ThreadPoolExecutor(max_workers=n_cores) as executor:
        results = list(tqdm(
            executor.map(get_fccs_for_reaction, ss_range, [reactions] * len(ss_range), [path] * len(ss_range)),
            total=len(ss_range),
            desc="Processing Steady States"
        ))

    # Combine all non-None results into a single dictionary
    for result in results:
        for reaction, df in result.items():
            if reaction not in results_dict:
                results_dict[reaction] = df
            else:
                results_dict[reaction] = pd.concat([results_dict[reaction], df], axis=1)

    return results_dict
