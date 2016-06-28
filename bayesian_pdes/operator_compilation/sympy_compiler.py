from bayesian_pdes import pairwise

import compilation_utils
from bayesian_pdes.operator_compilation.sympy_helpers import sympy_function, n_arg_applier


class OperatorSystem(object):
    def __init__(self, ops, ops_bar, compiled):
        self.operators = ops
        self.operators_bar = ops_bar
        self.__compiled_operators__ = compiled

    def __getitem__(self, item):
        if type(item) is not tuple:
            item = (item,)
        return self.__compiled_operators__[item]


def compile_sympy(operators, operators_bar, k, symbols, mode=None, sympy_function_kwargs=None, debug=False):
    ret = {}
    all_things = [()] + [(o,) for o in operators] + [(o,) for o in operators_bar] \
                 + [(o1,o2) for o1 in operators for o2 in operators_bar]
    for op in all_things:
        if op in ret:
            continue
        oped_kern = k
        for o in op:
            oped_kern = o(oped_kern)
        ret[op] = __functionize__(oped_kern, symbols, mode=mode, sympy_function_kwargs=sympy_function_kwargs, debug=debug)
    compilation_utils.infill_op_dict(operators, operators_bar, ret)
    return OperatorSystem(operators, operators_bar, ret)


# for now we only support sympy, maybe later support Theano?
# nest inside a function which will apply the result pairwise
def __functionize__(fun, symbols, mode=None, apply_factory=n_arg_applier, sympy_function_kwargs=None, debug=False):
    if sympy_function_kwargs is None:
        sympy_function_kwargs = {}
    sympy_fun = sympy_function(fun, symbols, mode=mode, apply_factory=apply_factory, debug=debug, **sympy_function_kwargs)

    def __ret_function(a, b, extra=None):
        return pairwise.apply(sympy_fun, a, b, extra)

    return __ret_function



