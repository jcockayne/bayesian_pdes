import numpy as np


def collocate(A, Abar, B, Bbar, k, symbols, interior_obs, boundary_obs):
    interior_points, interior_vals = interior_obs
    boundary_points, boundary_vals = boundary_obs

    # for now we only support sympy, maybe later support Theano?
    # nest inside a function which will apply the result pairwise
    def mf(fun):
        sympy_fun = sympy_function(fun, symbols)
        return lambda x_1, x_2: pairwise_apply(sympy_fun, x_1, x_2)

    k_eval = mf(k)
    A_k = mf(A(k))
    B_k = mf(B(k))
    Abar_k = mf(Abar(k))
    Bbar_k = mf(Bbar(k))

    AAbar_k = mf(A(Abar(k)))
    ABbar_k = mf(A(Bbar(k)))
    BAbar_k = mf(B(Abar(k)))
    BBbar_k = mf(B(Bbar(k)))

    # now, build the matrices we need
    Lbar = lambda x: np.c_[Abar_k(x, interior_points), Bbar_k(x, boundary_points)].T
    L = lambda x: np.c_[A_k(x,interior_points), B_k(x,boundary_points)].T
    LLbar = np.r_[
        np.c_[AAbar_k(interior_points,interior_points), ABbar_k(interior_points,boundary_points)],
        np.c_[BAbar_k(boundary_points,interior_points), BBbar_k(boundary_points,boundary_points)]
    ]
    LLbar_inv = np.linalg.inv(LLbar)

    # and the observation vector...
    g = np.concatenate([interior_vals, boundary_vals])

    # finally return the posterior
    def __posterior(test_points, samples):
        mu = Lbar(test_points).T.dot(LLbar_inv).dot(g)
        Sigma = k_eval(test_points, test_points) - Lbar(test_points).T.dot(LLbar_inv).dot(L(test_points))

        return np.random.multivariate_normal(mu, Sigma, samples)

    return __posterior


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
