from skimpy.utils import iterable_to_tabdict, TabDict
from skimpy.utils.namespace import *
from skimpy.utils.general import join_dicts
import pickle

from multiprocessing import Pool
def precompile_ode(kmodel, sim_type=QSSA, ncpu=1, expressions_file = None, path_to_so_file = None):

    # For security
    # kmodel.update()

    kmodel.sim_type = sim_type

    if not hasattr(kmodel, 'pool'):
        kmodel.pool = Pool(ncpu)

    # Recompile only if modified or simulation
    if kmodel._modified or kmodel.sim_type != sim_type:
        # Compile ode function
        ode_fun, variables = make_ode_fun(kmodel, sim_type, pool=kmodel.pool, expressions_file=expressions_file, path_to_so_file=path_to_so_file)


        # TODO define the init properly
        kmodel.ode_fun = ode_fun
        kmodel.variables = variables

        kmodel._modified = False
        kmodel._recompiled = True
        # Create initial_conditions from variables
        old_initial_conditions = kmodel.initial_conditions
        kmodel.initial_conditions = TabDict([(x,0.0) for x in kmodel.variables])
        # If data was stored previously in the initial conditions, recover it (needed for
        # serialization)
        kmodel.initial_conditions.update(old_initial_conditions)


def make_flux_fun_parallel(kinetic_model, sim_type, path_to_so_file=None):
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
    flux_fun = FluxFunction(variables, expr, all_parameters, pool=kinetic_model.pool, path_to_so_file=path_to_so_file, func_type='flux')
    flux_fun._parameter_values = {v:p.value for v,p in kinetic_model.parameters.items()}

    return flux_fun

def make_ode_fun(kinetic_model, sim_type, pool=None, custom_ode_update=None, expressions_file=None, path_to_so_file=None):
    """

    :param kinetic_model:
    :param sim_type:
    :return:
    """
    all_data = get_expressions_from_model(kinetic_model, sim_type)

    # get expressions for dxdt
    all_expr, _, all_parameters = list(zip(*all_data))

    # Flatten all the lists
    flatten_list = lambda this_list: [item for sublist in this_list \
                                      for item in sublist]

    all_parameters = flatten_list(all_parameters)
    all_parameters = list(set(all_parameters))
    all_parameters = iterable_to_tabdict(all_parameters, use_name=False)


    # Better since this is implemented now
    reactant_items = kinetic_model.reactants.items()
    variables = TabDict([(k,v.symbol) for k,v in reactant_items])

    #Compartments # CHECK IF THIS ONLY IS TRUE IF ITS NOT EMPTY
    if kinetic_model.compartments:
        #TODO Throw error if no cell reference compartment is given

        volume_ratios = TabDict([(k,v.compartment.parameters.cell_volume.symbol/
                            v.compartment.parameters.volume.symbol )
                           for k,v in kinetic_model.reactants.items()])
        for comp in kinetic_model.compartments.values():
            this_comp_parameters = {str(v.symbol):v.symbol for v in comp.parameters.values() }
            all_parameters.update( this_comp_parameters )
    else:
        volume_ratios = None
    
    # First check if the expressions have been calculated before but are not provided by the user
    if expressions_file is None:
        # Check if kin_logs folder exists
        if os.path.exists('kin_logs'):
            # Check if tmp_kmodel_expressions.pkl exists
            if os.path.exists('kin_logs/tmp_kmodel_expressions.pkl'):
                expressions_file = 'kin_logs/tmp_kmodel_expressions.pkl'
                print('Using symbolic expressions from kin_logs/tmp_kmodel_expressions.pkl')
        else:
            os.makedirs('kin_logs')

    # Either load the previous expressions or calculate them
    if expressions_file is not None:
        with open(expressions_file, 'rb') as f:
            expr = pickle.load(f)
    else:
        expr = make_expressions(variables,all_expr, volume_ratios=volume_ratios ,pool=pool)
        print('Symbolic Expressions saved to kin_logs/tmp_kmodel_expressions.pkl')
        with open('kin_logs/tmp_kmodel_expressions.pkl', 'wb') as f:
            pickle.dump(expr, f)

    # Apply constraints. Constraints are modifiers that act on
    # expressions
    for this_constraint in kinetic_model.constraints.values():
        this_constraint(expr)

    # NEW: Boundary conditions are now handled as parameters
    # Apply boundary conditions. Boundaries are modifiers that act on
    # expressions
    # for this_boundary_condition in kinetic_model.boundary_conditions.values():
    #     this_boundary_condition(expr)

    # Make vector function from expressions
    ode_fun = ODEFunction(kinetic_model, variables, expr, all_parameters, pool=pool,
                          custom_ode_update=custom_ode_update, path_to_so_file=path_to_so_file, func_type='ode')

    return ode_fun, variables

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

def make_expresson_single_var(input):
    var, all_flux_expr = input

    this_expr = dict()
    this_expr[var] = 0.0

    for this_reaction in all_flux_expr:
        for this_variable_key in this_reaction:
            if this_variable_key == var:
                this_expr[this_variable_key] += this_reaction[this_variable_key]

    return this_expr

def make_expressions(variables, all_flux_expr, volume_ratios=None,pool=None):

    if pool is None:
        expr = dict.fromkeys(variables.values(), 0.0)

        for this_reaction in all_flux_expr:
            for this_variable_key in this_reaction:
                try:
                    expr[this_variable_key] += this_reaction[this_variable_key]
                except KeyError:
                    pass

    else:

        inputs = [(v,all_flux_expr) for v in variables.values()]

        list_expressions = pool.map(make_expresson_single_var, inputs)

        expr = join_dicts(list_expressions)

    #Add compartment volumes
    if not volume_ratios is None:
        for k,v in variables.items():
            volume_ratio = volume_ratios[k]
            # Mutiply massbalance for each metabolite by volume ratio
            expr[v] = volume_ratio*expr[v]


    return expr


from numpy import array, double
from sympy import symbols, Symbol

from skimpy.utils.general import robust_index
from warnings import warn


class ODEFunction:
    def __init__(self, model, variables, expressions, parameters,
                 pool=None, with_time=False, custom_ode_update=None, path_to_so_file=None, func_type='ode'):
        """
        Constructor for a precompiled function to solve the ode epxressions
        numerically
        :param variables: a list of strings with variables names
        :param expressions: dict of sympy expressions for the rate of
                     change of a variable indexed by the variable name
        :param parameters: dict of parameters

        """
        self.variables = variables
        self.expressions = expressions
        self.model = model
        self.with_time = with_time
        self.custom_ode_update=custom_ode_update
        # Link to the model
        self._parameters = parameters

        # Unpacking is needed as ufuncify only take ArrayTypes
        the_param_keys = [x for x in self._parameters]
        the_variable_keys = [x for x in variables]

        if with_time:
            the_variable_keys = ['t',] + the_variable_keys

        sym_vars = list(symbols(the_variable_keys+the_param_keys))

        # Sort the expressions
        expressions = [self.expressions[x] for x in self.variables.values()]

        # Awsome magic
        self.function = make_cython_function(sym_vars, expressions, simplify=True, pool=pool, path_to_so_file=path_to_so_file, func_type=func_type)
    @property
    def parameters(self):
        model_params = self.model.parameters
        return TabDict((k, model_params[robust_index(k)].value)
                       for k in self._parameters)

    @parameters.setter
    def parameters(self, value):
        self._parameters = value

    def get_params(self):
        self._parameters_values = self.parameters.values()

    def __call__(self, t, y, ydot):
        if self.with_time:
            input_vars = [t,]+list(y)+list(self._parameters_values)
        else:
            input_vars = list(y)+list(self._parameters_values)
        self.function(input_vars, ydot)
        
        if not self.custom_ode_update is None:
            self.custom_ode_update( t, y, ydot)



import numpy as np
from sympy import symbols
from skimpy.utils.compile_sympy import make_cython_function


class FluxFunction:
    def __init__(self, variables, expr, parameters, pool=None, path_to_so_file=None, func_type='flux'):
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

        self.function = make_cython_function(sym_vars, expr.values(), simplify=True, pool=pool, path_to_so_file=path_to_so_file, func_type=func_type)


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




import ctypes
import re
import os

import numpy as np

import tempfile

import multiprocessing
from sympy.printing import ccode
from sympy import Symbol


#SUPPORTS ONLY GCC!
COMPILER = "gcc -fPIC -shared -w -O3"

# Test to write our own compiler
INCLUDE = "#include <stdlib.h>\n" \
          "#include <math.h>\n"

FUNCTION_DEFINITION_HEADER = "void function(double *input_array, double *output_array){ \n"
FUNCTION_DEFINITION_FOOTER = ";\n}"

def make_cython_function(symbols, expressions, quiet=True, simplify=True, optimize=False, pool=None, path_to_so_file=None, func_type=None):
    """
    Make a cython function from a list of sympy expressions
    :param symbols: list of sympy symbols
    :param expressions: list of sympy expressions
    :param quiet: print output from the compiler
    :param simplify: simplify the expressions before compilation
    :param pool: multiprocessing pool
    path_to_so_file: path to the so file if the function has been compiled before
    func_type: type of the function
    :return: a function that takes an array of input values and returns an array of output values
    """

    # First check if the so file exists but is not provided by the user
    if path_to_so_file is None and func_type is not None:
        # Check if kin_logs folder exists
        if os.path.exists('kin_logs'):
            # Check if tmp_kmodel_{func_type}_function.so exists
            if os.path.exists('kin_logs/tmp_kmodel_{}_function.so'.format(func_type)):
                path_to_so_file = 'kin_logs/tmp_kmodel_{}_function.so'.format(func_type)
                print('Using compiled function from kin_logs/tmp_kmodel_{}_function.so'.format(func_type))
        else:
            os.makedirs('kin_logs')

    if path_to_so_file is None:

        code_expressions = generate_vectorized_code(symbols,
                                                    expressions,
                                                    simplify=simplify,
                                                    pool=pool)


        # Write the code to a temp file
        code = INCLUDE + FUNCTION_DEFINITION_HEADER + code_expressions + FUNCTION_DEFINITION_FOOTER
        path_to_c_file = write_code_to_tempfile(code)
        path_to_so_file = path_to_c_file.replace('.c', '.so')

        # Compile the code
        cmd = " ".join([COMPILER, '-o ',path_to_so_file, path_to_c_file] )
        # Todo catch errors
        # Todo catch errors
        os.system(cmd)

        # Import the function
        fun = ctypes.CDLL(path_to_so_file)

        if func_type is not None:
            # copy the so file to kin_logs and rename it to tmp_kmodel_{func_type}_function.so
            os.system('cp {} kin_logs/'.format(path_to_so_file))
            os.system('mv kin_logs/{} kin_logs/tmp_kmodel_{}_function.so'.format(path_to_so_file.split('/')[-1], func_type))
            print('Compiled function saved to kin_logs/tmp_kmodel_{}_function.so'.format(func_type))
    
    else:
        fun = ctypes.CDLL(path_to_so_file)

    def this_function(input_array,output_array):
        # Input pointers
        fun.function.argtypes = [ctypes.POINTER(ctypes.c_double),
                                 ctypes.POINTER(ctypes.c_double),]
        #Cast to numpy float
        if not type(input_array) ==  np.ndarray.dtype:
            input_array = np.array(input_array, dtype=np.float)

        #x.ctypes.data_as(ctypes.POINTER(ctypes.c_long))
        fun.function(input_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) ,
                     output_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), )

    return this_function

def write_code_to_tempfile(code,file_path=None):
    if file_path is None:
        # make a tempfile
        (_, file_path) = tempfile.mkstemp(suffix = '.c')

    with open(file_path, "w") as text_file:
        text_file.write(code)
    return file_path

def generate_vectorized_code(inputs, expressions, simplify=True, optimize=False, pool=None):
    # input substitution dict:
    input_subs = {str(e): "input_array[{}]".format(i)
                  for i, e in enumerate(inputs)}

    if pool is None:
        cython_code = []
        for i,e in enumerate(expressions):
            if simplify:
                cython_code.append(generate_a_code_line_simplfied((i,e,input_subs)))
            else:
                cython_code.append(generate_a_code_line((i, e, input_subs)))

    else:
        if simplify:
            input_subs_input = [input_subs, ]*len(expressions)
            i,e =zip(*enumerate(expressions))
            cython_code = pool.map(generate_a_code_line_simplfied, zip(i,e,input_subs_input) )

        else:
            input_subs_input = [input_subs, ] * len(expressions)
            i, e = zip(*enumerate(expressions))
            cython_code = pool.map(generate_a_code_line, zip(i, e, input_subs_input))

    cython_code = ';\n'.join(cython_code)

    return cython_code


from sympy import cse

def generate_a_code_line_simplfied(input , optimize=False):
    i, e, input_subs = input

    # Use common sub expressions instead of simpilfy
    # Generate directly unique CSE Symbols and tranlate them to ccode
    if optimize:
        common_sub_expressions, main_expression = cse(e.simplify())
    else:
        common_sub_expressions, main_expression = cse(e)

    # Generate unique symbols for the common subexpressions
    cse_subs = {}
    common_sub_expressions_unique = []
    for this_cse in common_sub_expressions:
        gen_sym = str(this_cse[0])
        unique_sym = Symbol("cse_{}_{}".format(i,gen_sym))
        cse_subs[ this_cse[0]] = unique_sym
        common_sub_expressions_unique.append(
            [unique_sym,this_cse[1]] )

    # Substitute the cse symbols in mainexpression and other cse
    for this_cse in common_sub_expressions_unique:
        this_cse[1] = this_cse[1].subs(cse_subs)

    main_expression = main_expression[0].subs(cse_subs)

    cython_code = ''
    for this_cse in common_sub_expressions_unique:
        cython_code=cython_code+'double {} = {} ;\n'.format(str(this_cse[0]),
                                                    ccode(this_cse[1],standard='C99'))


    cython_code = cython_code+"output_array[{}] = {} ;".format(i,ccode(main_expression
                                                                      ,standard='C99')
                                                              )

    # Substitute integers in the cython code
    cython_code = re.sub(r"(\ |\+|[^e]\-|\*|\(|\)|\/|\,)([1-9])(\ |\+|\-|\*|\(|\)|\/|\,)",
                         r"\1 \2.0 \3 ",
                         cython_code)

    for str_sym, array_sym in input_subs.items():
        cython_code = re.sub(r"(\ |\+|\-|\*|\(|\)|\/|\,)({})(\ |\+|\-|\*|\(|\)|\/|\,)".format(str_sym),
                             r"\1 {} \3 ".format(array_sym),
                             cython_code)

    return cython_code


def generate_a_code_line(input, optimize=False):
    i, e, input_subs = input

    if optimize:
        cython_code = "output_array[{}] = {} ".format(i, ccode(e.simplify()), standard='C99')
    else:
        cython_code = "output_array[{}] = {} ".format(i,ccode(e, standard='C99'))


    # Substitute integers in the cython code
    cython_code = re.sub(r"(\ |\+|[^e]\-|\*|\(|\)|\/|\,)([1-9])(\ |\+|\-|\*|\(|\)|\/|\,)",
                         r"\1 \2.0 \3 ",
                         cython_code)

    for str_sym, array_sym in input_subs.items():
        cython_code = re.sub(r"(\ |\+|\-|\*|\(|\)|\/|\,)({})(\ |\+|\-|\*|\(|\)|\/|\,)".format(str_sym),
                             r"\1 {} \3 ".format(array_sym),
                             cython_code)
    return cython_code