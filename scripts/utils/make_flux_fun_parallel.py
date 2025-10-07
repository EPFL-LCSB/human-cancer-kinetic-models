from sympy import simplify

from skimpy.analysis.ode.ode_fun import ODEFunction
from skimpy.analysis.ode.flux_fun import FluxFunction
from skimpy.analysis.ode.gamma_fun import GammaFunction

from skimpy.utils import iterable_to_tabdict, TabDict
from skimpy.utils.namespace import *
from skimpy.utils.general import join_dicts


def get_expressions_from_model(kinetic_model, sim_type,
                               medium_symbols=None,
                               biomass_symbol=None):
    sim_type = sim_type.lower()
    # Get all variables and expressions (Better solution with types?)
    # TODO This should be a method in KineticModel that stores the expressions
    if sim_type == QSSA:
        all_data = []
        # TODO Modifiers should be applicable for all simulation types
        for this_reaction in kinetic_model.reactions.values():
            this_reaction.mechanism.get_qssa_rate_expression()
            # Update rate expressions
            for this_mod in this_reaction.modifiers.values():
                this_mod(this_reaction.mechanism.reaction_rates)
            this_reaction.mechanism.update_qssa_rate_expression()

            # Add modifier expressions
            for this_mod in this_reaction.modifiers.values():
                # Get parameters from modifiers
                for p_type, parameter in this_mod.parameters.items():
                    mod_sym = parameter.symbol
                    this_reaction.mechanism.expression_parameters.update([mod_sym])

                for r_type, reactant in this_mod.reactants.items():
                    # Add massbalances for modfier reactants if as non-zero stoich
                    if this_mod.reactant_stoichiometry[r_type] == 0:
                        continue

                    mod_sym = reactant.symbol
                    flux = this_reaction.mechanism.reaction_rates['v_net']
                    flux_expression = flux * this_mod.reactant_stoichiometry[r_type]
                    this_reaction.mechanism.expressions[mod_sym] = flux_expression

                    # Add small molecule parameters if they are
                    if reactant.type == PARAMETER:
                        this_reaction.mechanism.expression_parameters.update([mod_sym])

            flux = this_reaction.mechanism.reaction_rates['v_net']
            dxdt = this_reaction.mechanism.expressions
            parameters = this_reaction.mechanism.expression_parameters

            # For reactor building
            if not medium_symbols is None:
                vars_in_medium = [v for v in dxdt if v in medium_symbols]
                for v in vars_in_medium:
                    dxdt[v] = dxdt[v]*biomass_symbol

            all_data.append((dxdt, flux, parameters))

    elif sim_type == TQSSA:
        raise(NotImplementedError)

    elif sim_type == ELEMENTARY:
        all_data = []
        #TODO Modifiers sould be applicable for all simulation types
        for this_reaction in kinetic_model.reactions.values():
            this_reaction.mechanism.get_full_rate_expression()

            all_data.append((this_reaction.mechanism.expressions,
                             this_reaction.mechanism.reaction_rates,
                             this_reaction.mechanism.expression_parameters)
                            )
    else:
        raise(ValueError('Simulation type not recognized: {}'.format(sim_type)))

    return all_data

def make_flux_fun_parallel(kinetic_model, sim_type):
    """

    :param kinetic_model:
    :param sim_type:
    :return:
    """
    all_data = get_expressions_from_model(kinetic_model, sim_type)

    # Get flux expressions
    _, all_expr, all_parameters = list(zip(*all_data))

    reactions = kinetic_model.reactions.keys()

    # Flatten all the lists
    flatten_list = lambda this_list: [item for sublist in this_list \
                                      for item in sublist]

    if sim_type == ELEMENTARY:
        expr = [[(r+'_'+er, ex) for er, ex in e.items()] for r, e in zip(reactions, all_expr)]
        expr = TabDict(flatten_list(expr))
    else:
        expr = TabDict([(r, e) for r, e in zip(reactions, all_expr)])


    all_parameters = flatten_list(all_parameters)
    all_parameters = list(set(all_parameters))
    all_parameters = iterable_to_tabdict(all_parameters, use_name=False)

    # Better since this is implemented now
    reactant_items = kinetic_model.reactants.items()
    variables = TabDict([(k,v.symbol) for k,v in reactant_items])

    # Make vector function from expressions in this case all_expressions
    # are all the expressions indexed by the
    flux_fun = FluxFunction(variables, expr, all_parameters, pool=kinetic_model.pool)
    flux_fun._parameter_values = {v:p.value for v,p in kinetic_model.parameters.items()}

    return flux_fun


import numpy as np
from sympy import symbols
from skimpy.utils.compile_sympy import make_cython_function


class FluxFunction:
    def __init__(self, variables, expr, parameters, pool=None):
        """
        Constructor for a precompiled function to solve the ode epxressions
        numerically
        :param variables: a list of strings with variables names
        :param expr: dict of sympy expressions for the rate of
                     change of a variable indexed by the variable name
        :param parameters: dict of parameters with parameter values

        """
        self.variables = variables
        self.expr = expr
        self.parameters = parameters

        # Unpacking is needed as ufuncify only take ArrayTypes
        the_param_keys = [x for x in self.parameters]
        the_variable_keys = [x for x in variables]
        sym_vars = list(symbols(the_variable_keys+the_param_keys))

        self.function = make_cython_function(sym_vars, expr.values(), simplify=True, pool=pool)


    def __call__(self,concentrations,  parameters=None):
        # Todo handle different input types
        variables = [concentrations[str(x)] for x in self.variables]

        if parameters is None:
            input_vars = list(variables)+list(self.parameters.values())
        else:
            input_vars = list(variables) \
                         + [parameters[x] for x in self.parameters]

        fluxes = np.zeros(len(self.expr))

        self.function(input_vars, fluxes)

        return {k:v for k,v in zip(list(self.expr.keys()) , fluxes)}
