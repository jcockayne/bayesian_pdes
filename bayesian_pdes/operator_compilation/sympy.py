from bayesian_pdes.sympy_helpers import sympy_function, n_arg_applier
import pairwise
import numpy as np
import hashlib
import compilation_utils


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
    ret = {
        tuple(): __functionize(k, symbols, mode=mode, sympy_function_kwargs=sympy_function_kwargs)
    }
    for op in operators:
        if op not in ret:
            ret[(op,)] = __functionize(op(k), symbols, mode=mode, sympy_function_kwargs=sympy_function_kwargs)
    for op in operators_bar:
        if op not in ret:
            ret[(op,)] = __functionize(op(k), symbols, mode=mode, sympy_function_kwargs=sympy_function_kwargs)
    # combinations
    for op in operators:
        for op_bar in operators_bar:
            # don't do anything if already there
            if (op, op_bar) in ret:
                pass
            # exploit symmetry
            elif (op_bar, op) in ret:
                ret[(op, op_bar)] = ret[(op_bar, op)]
            # no choice!!
            else:
                ret[(op, op_bar)] = __functionize(op(op_bar(k)), symbols, mode=mode, sympy_function_kwargs=sympy_function_kwargs)
    compilation_utils.infill_op_dict(operators, operators_bar, ret)
    return OperatorSystem(operators, operators_bar, ret)


# for now we only support sympy, maybe later support Theano?
# nest inside a function which will apply the result pairwise
def __functionize(fun, symbols, mode=None, apply_factory=n_arg_applier, sympy_function_kwargs=None):
    if sympy_function_kwargs is None:
        sympy_function_kwargs = {}
    sympy_fun = sympy_function(fun, symbols, mode=mode, apply_factory=apply_factory, **sympy_function_kwargs)

    def __ret_function(a, b, extra=None):
        return pairwise.apply(sympy_fun, a, b, extra)

    return __ret_function



