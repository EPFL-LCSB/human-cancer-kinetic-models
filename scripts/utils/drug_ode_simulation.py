import numpy as np
import time
from skimpy.analysis.oracle.load_pytfa_solution import load_concentrations
from skimpy.utils.tabdict import TabDict
from skimpy.core.solution import ODESolution
from scikits.odes import ode
import pandas as pd
import configparser
import os    
import pickle

# Read scaling values from the config file
config = configparser.ConfigParser()
config_path = os.path.join(os.path.dirname(__file__), '../src/config.ini')
config.read(config_path)
scaling_section = 'scaling'
CONCENTRATION_SCALING = float(config[scaling_section].get('CONCENTRATION_SCALING', 1e6))
TIME_SCALING = float(config[scaling_section].get('TIME_SCALING', 1))
DENSITY = float(config[scaling_section].get('DENSITY', 1200))
GDW_GWW_RATIO = float(config[scaling_section].get('GDW_GWW_RATIO', 0.25))
flux_scaling_factor = 1e-3 * (GDW_GWW_RATIO * DENSITY) * CONCENTRATION_SCALING / TIME_SCALING

class ODESolution_prior:
    def __init__(self, conc, time):

        self.time    = np.array(time)

        self.species = np.array(conc)

        self.concentrations = conc


class ODESolution_post:
    def __init__(self, conc, time, model_ix):

        self.time    = np.array(time)

        self.species = np.array(conc)

        self.concentrations = conc
        self.model_ix = model_ix

class FluxSolution:
    def __init__(self, fluxes, time, model_ix):

        self.time    = np.array(time)

        self.fluxes = fluxes
        self.model_ix = model_ix

class ODESolution_ic50:
    def __init__(self, conc, time, model_ix, a_degradation):

        self.time    = np.array(time)

        self.species = np.array(conc)

        self.concentrations = conc
        self.model_ix = model_ix
        self.a_degradation = a_degradation

class CellViabilitySolution:
    def __init__(self, fluxes, time, model_ix, a_degradation):

        self.time    = np.array(time)

        self.fluxes = fluxes
        self.model_ix = model_ix
        self.a_degradation = a_degradation

def solver_ode(kmodel, time_out, solver_type='cvode', **kwargs):
    
    extra_options = {'old_api': False}
    kwargs.update(extra_options)
    

    kmodel.solver = ode(solver_type, kmodel.ode_fun, **kwargs)

    # Order the initial conditions according to variables
    ordered_initial_conditions = [kmodel.initial_conditions[variable]
                                  for variable in kmodel.variables]

    #Update fixed parameters
    kmodel.ode_fun.get_params()
    return kmodel.solver

def simulate_sample(ix, samples, tmodel, kmodel, parameter_population, CONCENTRATION_SCALING, targets,
                    TIME = np.linspace(0,35,120), time_limit = 1800, 
                    a_exponentrial_degradation = 0.1, k_exponentrial_degradation = 0.562, 
                    rtol=1e-6, atol=1e-6):
    
    print('Parameter sample: {}\n'.format(ix))

    # Get reference concentrations
    tfa_id, _ = ix.split(',')
    tfa_id = int(tfa_id)
    sample = samples.loc[tfa_id]
    init_concentrations = load_concentrations(sample, tmodel, kmodel,
                                              concentration_scaling=CONCENTRATION_SCALING)
    
    # Use a normalized enzyme concentration that will go from 1 to a_exponentrial_degradation with a rate of k_exponentrial_degradation
    for enz_name in targets:
        init_concentrations['E_' + enz_name] = 1

    # Load the parameter population and change the parameters connected to the targeted enzyme
    kmodel.parameters = parameter_population[ix]

    for enz_name in targets:
        # Add the exponential decay parameters
        kmodel.parameters['a_exponentrial_degradation_' + enz_name].value = a_exponentrial_degradation
        kmodel.parameters['k_exponentrial_degradation_' + enz_name].value = k_exponentrial_degradation
        # Replace the vmax with kcat*E
        kmodel.parameters['kcat_forward_' + enz_name].value = kmodel.parameters['vmax_forward_' + enz_name].value
        kmodel.parameters['vmax_forward_' + enz_name].value = None

    # Function to stop integration
    def rootfn(t, y, g, user_data):
        t_0 = user_data['time_0']
        t_max = user_data['max_time']
        t = time.time()
        if (t - t_0) >= t_max:
            g[0] = 0
            print('Did not converge in time')
        else:
            g[0] = 1

    t0 = time.time()
    user_data = {'time_0': t0,
                 'max_time': time_limit}  # provided in secomds

    # Create CVODE solver
    solver = solver_ode(kmodel,
                        TIME,
                        solver_type='cvode',
                        rtol=rtol,
                        atol=atol,
                        max_steps=1e6,
                        rootfn=rootfn,
                        nr_rootfns=1,
                        user_data=user_data
                        )

    ### ATTENTION: THE INITIAL CONC SHOULD BE GIVEN IN THE ORDER OF THE dX/dt IN THE ODE FUNCTION
    kmodel.initial_conditions = TabDict([(k, v) for k, v in init_concentrations.iteritems()])
    ordered_initial_conditions = [kmodel.initial_conditions[variable]
                                  for variable in kmodel.variables]
    solver.init_step(0, ordered_initial_conditions)

    # Solve for specific timeframe
    this_sol_qssa = solver.solve(TIME, ordered_initial_conditions)
    res = ODESolution(kmodel, this_sol_qssa)

    return [(res.time, res.concentrations), ix]

# This calculates the transient responses of the fluxes for each solution
def produce_flux_df(res, kmodel, parameter_population, TARGETS, calc_flux):
    ix = res.model_ix
    str_params = {k: v for k, v in parameter_population[ix].items()}

    for enz_name in TARGETS:
        str_params['kcat_forward_' + enz_name] = str_params['vmax_forward_' + enz_name]
        str_params['k_exponentrial_degradation_' + enz_name] = kmodel.parameters['k_exponentrial_degradation_' + enz_name].value
        str_params['a_exponentrial_degradation_' + enz_name] = kmodel.parameters['a_exponentrial_degradation_' + enz_name].value

    flux_df = []
    for _, val in res.concentrations.iterrows():
        fluxes = pd.Series(calc_flux(val, parameters=str_params))
        flux_df.append(fluxes/flux_scaling_factor)
        
    return ix, pd.DataFrame(flux_df)


def run_simulation_ic50(args, filename, samples, tmodel, kmodel, parameter_population, CONCENTRATION_SCALING, 
                        targets, TIME = np.linspace(0,35,120), time_limit = 1800, 
                        a_exponentrial_degradation = 0.1, k_exponentrial_degradation = 0.562,  rtol=1e-6, atol=1e-6):
    ix, a = args
    # Check if the file already exists
    if os.path.exists(filename):
        print(f"File {filename} already exists. Skipping simulation for {ix} with a={a}.")
        return False

    res = simulate_sample(ix, samples, tmodel=tmodel, kmodel=kmodel, 
                           parameter_population=parameter_population, CONCENTRATION_SCALING=CONCENTRATION_SCALING,
                           targets=targets, TIME=TIME, time_limit=time_limit,
                           a_exponentrial_degradation=a_exponentrial_degradation, k_exponentrial_degradation=k_exponentrial_degradation,
                           rtol=rtol, atol=atol)
    
    with open(filename, 'wb') as f:
        pickle.dump(res, f)
    return True


from sympy import Symbol
def produce_biomass_df(res, kmodel, targets, parameter_population):
    a_degradation = res.a_degradation
    model_id = res.model_ix
    str_params = {Symbol(k): v for k, v in parameter_population[model_id].items()}

    # for enz_name in targets:
    #     str_params[Symbol('kcat_forward_' + enz_name)] = str_params[Symbol('vmax_forward_' + enz_name)]
    #     str_params[Symbol('k_exponentrial_degradation_' + enz_name)] = kmodel.parameters['k_exponentrial_degradation_' + enz_name].value
    #     str_params[Symbol('a_exponentrial_degradation_' + enz_name)] = a_degradation

    new_expr = kmodel.reactions.biomass.mechanism.reaction_rates['v_net']
    biomass_expr = new_expr.xreplace(str_params)
    # Calculate the biomass fluxes
    flux_df = []
    for _, val in res.concentrations.iterrows():
        biomass = biomass_expr.xreplace({Symbol(k): v for k, v in val.items()})/flux_scaling_factor
        flux_df.append(float(biomass))
    
    del biomass_expr, str_params

        
    return model_id, a_degradation, pd.DataFrame(flux_df, columns=['biomass'])