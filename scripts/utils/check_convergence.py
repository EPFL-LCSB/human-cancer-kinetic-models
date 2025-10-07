from tqdm.auto import tqdm
import pandas as pd

def check_convergence(fcc_rxn_slice, latest_ss_index, tolerance=0.01, moment_number=3, insignificant_effect = 1e-4):
    
    '''Check if the moments of each FCC have converged simce the previous steady state'''

    converged = True

    old_columns = [col for col in fcc_rxn_slice.columns if not col.startswith(latest_ss_index)]
    if len(old_columns) == len(fcc_rxn_slice.columns):
        raise ValueError('The steady state number does not belong to the FCC dataframe...')

    for par_name in tqdm(fcc_rxn_slice.index, desc='Checking convergence...'):
        data_series_total = fcc_rxn_slice.loc[par_name].dropna()
        data_series_old = fcc_rxn_slice.loc[par_name, old_columns].dropna()

        # Skip if the FCC has an average absolute value close to zero
        if abs(data_series_total.mean()) < insignificant_effect:  # IMPORTANT: THIS CRITERIA IS ARBITRARY AND SHOULD BE ADJUSTED IF NEED BE
            print('The average of {} is close to zero, skipping...'.format(par_name))
            continue
        
        # Calculate the average before and after this steady state
        avg_old = data_series_old.mean()
        avg_total = data_series_total.mean()
        rel_abs_diff = abs(avg_total - avg_old) / abs(avg_old)
        if rel_abs_diff > tolerance:
            converged = False
            print('The average of {} has not converged (relative abs diff: {:.3f})'.format(par_name, rel_abs_diff))
            break

        if moment_number > 1:
            # Calculate the variance before and after this steady state
            var_old = data_series_old.var()
            var_total = data_series_total.var()
            rel_abs_diff = abs(var_total - var_old) / abs(var_old)
            if rel_abs_diff > tolerance:
                converged = False
                print('The variance of {} has not converged (relative abs diff: {:.3f})'.format(par_name, rel_abs_diff))
                break
        
        if moment_number > 2:
            # Calculate the skewness before and after this steady state
            skew_old = data_series_old.skew()
            skew_total = data_series_total.skew()
            rel_abs_diff = abs(skew_total - skew_old) / abs(skew_old)
            if rel_abs_diff > tolerance:
                converged = False
                print('The skewness of {} has not converged (relative abs diff: {:.3f})'.format(par_name, rel_abs_diff))
                break
    return converged