"""
Code to run variability analysis for a specified variable
"""
from skimpy.analysis.oracle import *  # Important to avoid MinFluxVariable serialization error
from pytfa.io.json import load_json_model
from skimpy.io.yaml import load_yaml_model
from skimpy.utils.general import sanitize_cobra_vars
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sys import argv
from pytfa.analysis.variability import _variability_analysis_element


def element_variability_analysis(tmodel, varname, kind=None):
    if kind == 'reactions':
        print('MinMax for variable ' + varname)
        rxn = tmodel.reactions.get_by_id(varname)
        tva_min = _variability_analysis_element(tmodel, rxn.forward_variable-rxn.reverse_variable, 'min')
        tva_max = _variability_analysis_element(tmodel, rxn.forward_variable-rxn.reverse_variable, 'max')
        tmodel.objective = tmodel.reactions.biomass  # change to the model's original objective function
        return [tva_min, tva_max]
    else:
        print('MinMax for variable '+varname)
        tva_min = _variability_analysis_element(tmodel, tmodel.variables[varname], 'min')
        tva_max = _variability_analysis_element(tmodel, tmodel.variables[varname], 'max')
        tmodel.objective = tmodel.reactions.biomass  # change to the model's original objective function
        return [tva_min, tva_max]
