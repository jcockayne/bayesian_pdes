import numpy as np


def collocate(operators, operators_bar, k, symbols, observations):
    """
    Construct a collocation approximation to the system of operators supplied.
    :param operators: List of operators operators (as functions which operate on sympy expressions)
    :param operators_bar: List of operators working on the second argument of the kernel.
    :param k: The kernel itself, as a sympy expression
    :param symbols: Sympy symbols for the kernel. Note that thanks to sympy being a bit peculiar we can't easily use
    vector operations here - instead each component must be supplied separately if we are working in a higher-dimensional
    space.
    In this event, the sympy symbols should be grouped into sub-lists according to which vector they belong to,
    eg. [[x_1, x_2], [y_1, y_2]]
    :param observations: List of observations. Size should match the size of the list of operators.
    It should be a list of tuples, [(obs locations, obs values)], where obs locations has m rows and d columns and
    values has m rows and 1 column. Here m is the number of points observed and d is the dimension of each point.
    :return: A function which, when given a set of test points, returns the posterior mean and covariance at those
    points.
    """

    k_eval = functionize(k, symbols)

    # build the 1D operators
    def apply_1d(this_operators):
        tmp = []
        for op in this_operators:
            tmp.append(functionize(op(k), symbols))
        return lambda x: np.concatenate([f(x, obs[0]) for f, obs in zip(tmp, observations)], 1).T

    L = apply_1d(operators)
    Lbar = apply_1d(operators_bar)

    LLbar = calc_LLbar(operators, operators_bar, k, symbols, observations)
    LLbar_inv = np.linalg.inv(LLbar)

    # and the observation vector...
    g = np.concatenate([val for _, val in observations])

    # finally return the posterior
    def __posterior(test_points):
        mu = Lbar(test_points).T.dot(LLbar_inv).dot(g)
        Sigma = k_eval(test_points, test_points) - Lbar(test_points).T.dot(LLbar_inv).dot(L(test_points))

        return mu, Sigma

    return __posterior


# for now we only support sympy, maybe later support Theano?
# nest inside a function which will apply the result pairwise
def functionize(fun, symbols):
    sympy_fun = sympy_function(fun, symbols)
    return lambda x_1, x_2: pairwise_apply(sympy_fun, x_1, x_2)


def pairwise_apply(fun, A, B):
    """
    Create a gram matrix from applying a scalar function pairwise to vectors from A and B
    :param fun: The function. Should take two vectors and return a scalar.
    :param A: A matrix whose columns represent dimensions, and whose rows represent observations.
    :param B: Same specification as A.
    :return: The Gram Matrix whose (i,j) element is fun(A[i,:], B[j,:])
    """
    # TODO: slow!
    ret = np.empty((A.shape[0], B.shape[0]))
    for i in xrange(A.shape[0]):
        for j in xrange(B.shape[0]):
            ret[i,j] = fun(A[i,:], B[j,:])
    return ret


def calc_LLbar(operators, operators_bar, k, symbols, observations):
    # and build the 2D matrix
    LLbar = []
    for op, obs_1 in zip(operators, observations):
        tmp = []
        for op_bar, obs_2 in zip(operators_bar, observations):
            points_1, _ = obs_1
            points_2, _ = obs_2

            fun_op = functionize(op(op_bar(k)), symbols)
            applied = fun_op(points_1, points_2)
            tmp.append(applied)
        LLbar.append(np.concatenate(tmp, 1))

    return np.concatenate(LLbar, 0)


def sympy_function(sympy_expression, sympy_symbols):
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
