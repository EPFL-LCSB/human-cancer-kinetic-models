from concurrent.futures import ProcessPoolExecutor
import numpy as np
from skimpy.utils.general import sanitize_cobra_vars
from skimpy.utils.conversions import deltag0_to_keq
from skimpy.core.parameters import ParameterValues

import pandas as pd

def load_equilibrium_constants(solution_raw, tmodel, kmodel,
                               concentration_scaling=None,
                               in_place=False,
                               num_cpus=1):
    if concentration_scaling is None:
        raise ValueError("concentration_scaling is required as input or field of kmodel")

    equilibrium_constant_dict = dict()
    RT = tmodel.RT
    rnxs_ids_with = {dg.id for dg in tmodel.delta_g}

    # Function to process a single reaction
    def process_reaction(pytfa_rxn):
        if pytfa_rxn.id not in kmodel.reactions or pytfa_rxn.id not in rnxs_ids_with:
            return None

        deltag0 = solution_raw[tmodel.delta_g.get_by_id(pytfa_rxn.id).name]
        for met, stoich in pytfa_rxn.metabolites.items():
            kin_met_id = sanitize_cobra_vars(met)
            if kin_met_id in kmodel.reactants or kin_met_id in kmodel.parameters:
                var_met_lc = tmodel.log_concentration.get_by_id(met.id).name
                met_lc = solution_raw[var_met_lc]
                deltag0 -= stoich * RT * (met_lc + np.log(concentration_scaling))

        try:
            k_eq = kmodel.reactions[pytfa_rxn.id].parameters['k_equilibrium']
        except KeyError:
            return None

        k_eq_value = deltag0_to_keq(deltag0, tmodel.TEMPERATURE, gas_constant=tmodel.GAS_CONSTANT)
        if in_place:
            k_eq.value = k_eq_value
        return k_eq.symbol, k_eq_value

    # Parallel execution of reaction processing
    with ProcessPoolExecutor(max_workers=num_cpus) as executor:
        results = executor.map(process_reaction, tmodel.reactions)

    # Collect results
    for result in results:
        if result is not None:
            symbol, value = result
            equilibrium_constant_dict[symbol] = value

    return ParameterValues(equilibrium_constant_dict, kmodel)
