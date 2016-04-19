try:
    from autograd import numpy as np
except:
    print("Autograd not available; using standard numpy.")
    import numpy as np

import operator_compilation


def collocate(operators, operators_bar, k, symbols, observations, op_cache=None, fun_args=None):
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
    for locs, vals in observations:
        if vals is None: continue
        err = " (Loc has shape {}, vals have shape {})".format(locs.shape, vals.shape)
        if locs.shape[0] != vals.shape[0]:
            raise Exception("Number of obs not consistent with location of obs " + err)
        if len(locs.shape) != 2:
            raise Exception("Obs locations must be two-dimensional " + err)

    if len(operators) != len(observations):
        raise Exception('Number of obs not consistent with number of operators ({} observations but {} operators)'
                        .format(len(observations), len(operators)))

    if op_cache is None:
        op_cache = operator_compilation.compile_sympy(operators, operators_bar, k, symbols)

    LLbar = calc_LLbar(operators, operators_bar, observations, op_cache, fun_args)
    LLbar_inv = np.linalg.inv(LLbar)

    # finally return the posterior
    return CollocationPosterior(operators, operators_bar, op_cache, observations, LLbar_inv, fun_args)


def calc_LLbar(operators, operators_bar, observations, op_cache, fun_args=None):
    # and build the 2D matrix
    LLbar = []

    for op, obs_1 in zip(operators, observations):
        tmp = []
        for op_bar, obs_2 in zip(operators_bar, observations):
            points_1, _ = obs_1
            points_2, _ = obs_2

            fun_op = op_cache[(op, op_bar)]

            applied = fun_op(points_1, points_2, fun_args)
            tmp.append(applied)
        LLbar.append(np.concatenate(tmp, axis=1))

    return np.concatenate(LLbar)


def calc_side_matrices(operators, operators_bar, obs, test_points, op_cache, fun_args=None):
    obs_points = np.r_[[p for p, _ in obs]]
    L = []
    Lbar = []
    for op, op_bar, point in zip(operators, operators_bar, obs_points):
        f = op_cache[(op,)]
        fbar = op_cache[(op_bar,)]
        L.append(f(point, test_points, fun_args))
        Lbar.append(fbar(test_points, point, fun_args))
    L = np.concatenate(L)
    Lbar = np.concatenate(Lbar, axis=1)
    return L, Lbar


class CollocationPosterior(object):
    def __init__(self, operators, operators_bar, op_cache, obs, LLbar_inv, fun_args=None):
        self.__operators = operators
        self.__operators_bar = operators_bar
        self.__op_cache = op_cache
        self.__obs = obs
        self.__LLbar_inv = LLbar_inv
        self.__fun_args = fun_args

    def __call__(self, test_points):
        return self.posterior(test_points)

    def posterior(self, test_points):
        g = np.concatenate([val for _, val in self.__obs])
        mu_multiplier, Sigma = self.no_obs_posterior(test_points)
        return np.dot(mu_multiplier, g), Sigma

    def mean(self, test_points, g=None):
        if g is None:
            g = np.concatenate([val for _, val in self.__obs])
        L, Lbar = calc_side_matrices(self.__operators, self.__operators_bar, self.__obs, test_points, self.__op_cache, self.__fun_args)
        mu_multiplier = np.dot(Lbar, self.__LLbar_inv)

        return np.dot(mu_multiplier, g)

    def no_obs_posterior(self, test_points):
        L, Lbar = calc_side_matrices(self.__operators, self.__operators_bar, self.__obs, test_points, self.__op_cache, self.__fun_args)

        mu_multiplier = np.dot(Lbar, self.__LLbar_inv)

        k_eval = self.__op_cache[()]
        k_mat = k_eval(test_points, test_points, self.__fun_args)
        Sigma = k_mat - np.dot(mu_multiplier, L)

        return mu_multiplier, Sigma

    def diagonal_covariance(self, test_points):
        ret = np.empty((test_points.shape[0], 1))
        for i in xrange(test_points.shape[0]):
            _, cov = self.no_obs_posterior(test_points[i, :][None, :])
            ret[i, 0] = cov
        return ret