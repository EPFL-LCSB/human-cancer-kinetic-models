from scikits.odes import ode
import numpy as np
from skimpy.analysis.oracle.load_pytfa_solution import load_concentrations
from skimpy.analysis.ode import sample_initial_concentrations
import configparser
import os
import tqdm
from skimpy.utils.tabdict import TabDict
from skimpy.core.solution import ODESolutionPopulation
from skimpy.core.solution import ODESolution

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


def perturb_ode_solver(ix, tmodel, kmodel, samples, parameter_population, tfa_id, path_to_sol,
                       RELATIVE_CHANGE = 0.1, N_SAMPLES = 100, TIME = np.logspace(-9, np.log10(60), 10),
                       TOTAL_TIME_STEPS = 10,
                       rtol= 1e-9, atol=1e-9):
    # print('Begining perturbations for parameter set {} out of {}'.format(list(parameter_population._index.keys()).index(ix), len(parameter_population._index.keys())))
    # Get reference concentrations
    tfa_id = int(tfa_id)
    sample = samples.loc[tfa_id]
    reference_concentrations = load_concentrations(sample, tmodel, kmodel,
                                                   concentration_scaling=CONCENTRATION_SCALING)
    # Add parameters
    kmodel.parameters = parameter_population[str(ix)]

    # Add a perturbation vector N_SAMPLES times to the reference concentrations
    # to create N_SAMPLES perturbed initial conditions
    perturbed_concentrations = sample_initial_concentrations(kmodel,
                                                             reference_concentrations,
                                                             lower_bound=1-RELATIVE_CHANGE,
                                                             upper_bound=1+RELATIVE_CHANGE,
                                                             n_samples=N_SAMPLES)

    solutions = []

    # Loop through all perturbation vectors, and study their evolution in time
    for i, this_perturbations in tqdm(perturbed_concentrations.iterrows(), total=N_SAMPLES):
        print(ix,i)

        import time 

        def rootfn(t, y, g, user_data):
            # print(t)
            t_0 = user_data['time_0']
            t_max = user_data['max_time']
            t = time.time()

            y0 = user_data['y0']
            n_max = user_data['max_norm']
            norm = np.sqrt(np.square((y - y0) / y0).sum())
            if (t - t_0) >= t_max:
                g[0] = 0
                print('Did not converge in time')
            elif norm >= n_max:
                g[0] = 0
                print('The model diverged')
            else:
                g[0] = 1

        t0 = time.time()
        user_data = {'y0': reference_concentrations[kmodel.variables].values,
                    'max_norm': 1e5,
                    'time_0': t0,
                    'max_time': 1200}

        # Integrate perturbations
        kmodel.initial_conditions = TabDict([(k, v) for k, v in this_perturbations.iteritems()])
        ordered_initial_conditions = [kmodel.initial_conditions[variable]
                                for variable in kmodel.variables]
        solver = solver_ode(kmodel, TIME, solver_type='cvode', rtol=rtol, atol=atol, max_steps=1e9, rootfn=rootfn, nr_rootfns=1, user_data=user_data)
        solver.init_step(0, ordered_initial_conditions)

        # Solve for specific timeframe
        this_sol_qssa = solver.solve(TIME, ordered_initial_conditions)
        this_sol_qssa = ODESolution(kmodel, this_sol_qssa)
        
        solutions.append(this_sol_qssa)
    

    damped_time = TOTAL_TIME_STEPS  # calculate the damping effect after one cell cylce
    min_thresh = 0.05  # The dumping effect cutoff value

    # Load the steady state values
    variables = this_sol_qssa.concentrations.columns
    steady_state_values = reference_concentrations[variables]
    results = []
    # If the model achieves 95% damping of the original perturbation then it is considered (non_linearly) stable
    for sol in solutions:
        diff = 1-np.abs((sol.concentrations.iloc[damped_time])/steady_state_values)
        converge_norm = np.linalg.norm(diff, ord=np.inf)
        
        # Check the results after 3 doubling times
        final_diff = 1-np.abs((sol.concentrations.iloc[-1])/steady_state_values)
        final_converge_norm = np.linalg.norm(final_diff, ord=np.inf)

        if converge_norm <= min_thresh:
            # Model was both fast and stable
            results.append(1)
        elif final_converge_norm <= min_thresh:
            # Model was stable but not fast
            results.append(2)
        else:
            # Model was unstable
            results.append(0)

    solpop = ODESolutionPopulation(solutions)
    solpop.data.to_csv(path_to_sol)
    return solpop, results, steady_state_values