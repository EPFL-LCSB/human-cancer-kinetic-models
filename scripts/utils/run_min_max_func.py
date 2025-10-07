from pytfa.io.json import load_json_model
from pytfa.optim.variables import DeltaG, DeltaGstd, ThermoDisplacement, LogConcentration
from pytfa.analysis import variability_analysis
from skimpy.analysis.oracle.minimum_fluxes import MinFLux, \
    MinFLuxVariable
import multiprocessing as mp

def run_min_max(kind, tmodel_):
    ll = variability_analysis(tmodel_, kind=kind)
    return [ll]