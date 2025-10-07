import matplotlib.pyplot as plt
import numpy as np
import time
from scikits.odes import ode
from skimpy.analysis.oracle.load_pytfa_solution import load_concentrations
from skimpy.utils.tabdict import TabDict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimpy.core.solution import ODESolution
import time


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


def simulate_sample(ix, samples, tmodel, kmodel, parameter_population, 
                    CONCENTRATION_SCALING, time_limit=3600, a_degradation=0.1, 
                    k_degradation=0.562, enzyme_concentration=1, time_points=120, t_span=35, solver_type='cvode', 
                    rtol=1e-6, atol=1e-6, max_steps=1e6):
    """
    Simulate a sample using the provided kinetic model and parameters.

    Parameters:
    ix (str): Identifier for the parameter sample.
    samples (pd.DataFrame): DataFrame containing the samples.
    tmodel_ (object): Thermodynamic model.
    kmodel_ (object): Kinetic model.
    parameter_population (ParameterValuePopulation): Population of parameter values.
    CONCENTRATION_SCALING (float): Scaling factor for concentrations.
    time_limit (int): Time limit for the simulation in seconds.
    a_degradation (float): Degradation rate constant 'a'.
    k_degradation (float): Degradation rate constant 'k'.
    enzyme_concentration (float): Initial concentration of the enzyme.
    time_points (int): Number of time points for the simulation.
    t_span (float): Time span for the simulation.
    solver_type (str): Type of solver to use.
    rtol (float): Relative tolerance for the solver.
    atol (float): Absolute tolerance for the solver.
    max_steps (int): Maximum number of steps for the solver.

    Returns:
    list: A list containing the simulation results and the sample identifier.
    """
    print('Parameter sample: {}'.format(ix))

    # Get reference concentrations
    tfa_id, _ = ix.split(',')
    tfa_id = int(tfa_id)
    sample = samples.loc[tfa_id]
    init_concentrations = load_concentrations(sample, tmodel, kmodel, concentration_scaling=CONCENTRATION_SCALING)
    init_concentrations['E_TMDS'] = enzyme_concentration

    kmodel.parameters = parameter_population[ix]
    kmodel.parameters.a_degradation.value = a_degradation
    kmodel.parameters.k_degradation.value = k_degradation
    kmodel.parameters.kcat_forward_TMDS.value = kmodel.parameters.vmax_forward_TMDS.value
    kmodel.parameters.vmax_forward_TMDS.value = None

    TIME = np.linspace(0, t_span, time_points)

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
    user_data = {'time_0': t0, 'max_time': time_limit}  # provided in minutes in seconds

    # Create CVODE solver
    solver = solver_ode(kmodel, TIME, solver_type=solver_type, rtol=rtol, atol=atol, max_steps=max_steps, 
                        rootfn=rootfn, nr_rootfns=1, user_data=user_data)

    # Set initial conditions
    kmodel.initial_conditions = TabDict([(k, v) for k, v in init_concentrations.iteritems()])
    ordered_initial_conditions = [kmodel.initial_conditions[variable] for variable in kmodel.variables]
    solver.init_step(0, ordered_initial_conditions)

    # Solve for specific timeframe
    this_sol_qssa = solver.solve(TIME, ordered_initial_conditions)
    res = ODESolution(kmodel, this_sol_qssa)
    return [(res.time, res.concentrations), ix]


def plot_trajectories(reaction_df, samples_picked, time, t_span, save_path=None):
    """
    Plots the trajectories of reactions over time for different clusters of samples.
    Parameters:
    reaction_df (pd.DataFrame): DataFrame containing the reaction data to be plotted.
    samples_picked (pd.DataFrame): DataFrame containing the samples and their corresponding cluster groups.
    time (np.ndarray): Array of time to be considered for plotting.
    t_span (float): The time span up to which the trajectories should be plotted.
    save_path (str, optional): Path to save the plot. If None, the plot will be displayed but not saved.
    Returns:
    None
    """
    
    # Normalize the data with respect to the first point
    normalized_df = reaction_df.div(reaction_df.iloc[0])

    # Define a colormap
    cmap = plt.cm.viridis(np.linspace(0, 1, len(samples_picked.group.unique())))
    
    # Initialize a figure
    plt.figure(figsize=(12, 8))
    legend_handles = []

    # Subset the time points based on t_span
    time_indices = time <= t_span
        # Plot the trajectories for each cluster
    for i, group_id in enumerate(samples_picked.group.unique()):
        color = cmap[i]  # Get a unique color for each cluster

        steady_states = samples_picked.loc[samples_picked.group == group_id].index
        models = [model for model in reaction_df.columns if int(model.split(',')[0]) in steady_states]

        # Extract the data for the current cluster
        cluster_data = normalized_df.loc[:, models]

        # Calculate the average trajectory for the cluster
        average_trajectory = cluster_data.mean(axis=1)

        # Calculate the standard deviation for the cluster
        std_deviation = cluster_data.std(axis=1)

        # Find the trajectory closest to the average
        distances = cluster_data.apply(lambda col: np.linalg.norm(col - average_trajectory), axis=0)
        closest_trajectory = cluster_data.loc[:, distances.idxmin()]

        # Plot the closest trajectory
        plt.plot(time[time_indices], closest_trajectory[time_indices], color=color, label=f'Cluster {i+1}', linewidth=2, alpha=0.5)

        # Plot the error bounds (one standard deviation)
        # plt.fill_between(res.time[time_indices], 
        #                  average_trajectory[time_indices] - std_deviation[time_indices], 
        #                  average_trajectory[time_indices] + std_deviation[time_indices],
        #                  color=color, alpha=0.01)

        # Add legend handles
        legend_handles.append(plt.Line2D([0], [0], color=color, lw=2, label=f'Cluster {i+1}'))

    # Add legend to the plot
    plt.legend(handles=legend_handles)
    plt.xlabel('Time')
    plt.ylabel('Normalized Flux')
    
    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path)
    plt.show()