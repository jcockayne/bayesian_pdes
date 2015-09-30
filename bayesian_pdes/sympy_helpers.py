import numpy as np


def sympy_function(sympy_expression, sympy_symbols):
    """
    Convert a sympy expression into a function whose arguments reflect a particular vector structure.
    :param sympy_expression: A sympy expression in a flattened set of variables.
    :param sympy_symbols: The symbols from the expression, assembled into arrays reflecting their vector structure.
    Examples:
    [[x_1, x_2], [y_1, y_2]] : expects two vectors (iterables) of length 2
    [[x_1, x_2], y] : expects a vector of length 2 and a scalar.
    :return: The callable expression
    """
    from sympy import lambdify
    flattened = []
    for s in sympy_symbols:
        if type(s) is list:
            flattened += s
        else:
            flattened.append(s)
    sympyd = lambdify(flattened, sympy_expression)

    # wrap inside a function which will do the same flattening again, checking shapes as it goes.
    def __apply_to_stuff(*args):
        assert len(args) == len(sympy_symbols), "Received {} args but expected {}".format(len(args), len(sympy_symbols))
        flattened = []
        for ix, arg, symb in zip(range(len(args)), args, sympy_symbols):
            if type(symb) is list:
                assert type(arg) in [np.ndarray, list], "Expected an iterable for arg {} but received {}".format(ix, type(arg))
                assert len(symb) == len(arg), 'Argument {} has wrong length. Received {} but expected {}'.format(ix, len(arg), len(symb))
                flattened += list(arg)
            # todo: could do with checks here as well.
            else:
                flattened.append(arg)

        return sympyd(*flattened)

    return __apply_to_stuff