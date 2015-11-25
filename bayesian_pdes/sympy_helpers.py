import numpy as np


def n_arg_applier(compiled_func, symbols):
    """
    Apply the function to the arguments assuming there are two arguments that need to be flattened.
    This is much faster but obviously if there _aren't_ two args it won't work!
    :param compiled_func: The sympy function.
    :param symbols: The sympy symbols
    :return: A flattened function applicator
    """
    # this is unsafe!
    def __apply_two_arg(*args):
        ret = compiled_func(*np.concatenate(args))
        return ret

    return __apply_two_arg


def generic_applier(compiled_func, symbols):
    """
    Generically apply the function to the arguments, flattening into the same format.
    This is slow but should always work.
    :param compiled_func: The sympy function.
    :param symbols: The sympy symbols
    :return: A flattened function applicator
    """
    # wrap inside a function which will do the same flattening again, checking shapes as it goes.
    def __apply_generic(*args):
        assert len(args) == len(symbols), "Received {} args but expected {}".format(len(args), len(symbols))
        flattened = []

        # import time
        # start = time.time()
        for ix, arg, symb in zip(range(len(args)), args, symbols):
            if type(symb) is list:
                assert type(arg) in [np.ndarray, list], "Expected an iterable for arg {} but received {}".format(ix, type(arg))
                assert len(symb) == len(arg), 'Argument {} has wrong length. Received {} but expected {}'.format(ix, len(arg), len(symb))
                flattened += list(arg)
            # todo: could do with checks here as well.
            else:
                flattened.append(arg)
        # print('Flattened args,{}'.format(time.time() - start))
        # start = time.time()
        ret = compiled_func(*flattened)
        # print('Evaluate function,{}'.format(time.time() - start))
        return ret

    return __apply_generic


def sympy_function(sympy_expression, sympy_symbols, mode=None, apply_factory=generic_applier):
    """
    Convert a sympy expression into a function whose arguments reflect a particular vector structure.
    :param sympy_expression: A sympy expression in a flattened set of variables.
    :param sympy_symbols: The symbols from the expression, assembled into arrays reflecting their vector structure.
    Examples:
    [[x_1, x_2], [y_1, y_2]] : expects two vectors (iterables) of length 2
    [[x_1, x_2], y] : expects a vector of length 2 and a scalar.
    :param mode: Either 'lambda' or 'compile'. 'lambda' will use sympy.lambdify while 'compile' will use sympy.autowrap
    default: Lambda
    :param apply_factory: An expression which will apply the flattening operator to the arguments, and then pass
    to the sympy expression.
    :return: The callable expression
    """
    from sympy import lambdify
    from sympy.utilities import autowrap
    flattened = []
    for s in sympy_symbols:
        if type(s) is list:
            flattened += s
        else:
            flattened.append(s)
    if mode is None or mode.lower() == 'lambda':
        sympyd = lambdify(flattened, sympy_expression)
    elif mode.lower() == 'compile':
        sympyd = autowrap.autowrap(sympy_expression, args=flattened)
    else:
        raise Exception('Mode for generation of sympy function {} not understood.'.format(mode))

    return apply_factory(sympyd, sympy_symbols)