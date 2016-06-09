try:
    from autograd import numpy as np
except:
    print("Autograd not available; using standard numpy.")
    import numpy as np

import logging

logger = logging.getLogger(__name__)


def collocate(operators, operators_bar, observations, op_system, fun_args=None):
    """
    Construct a collocation approximation to the system of operators supplied.
    :param operators: List of operators operators (as functions which operate on sympy expressions)
    :param operators_bar: List of operators working on the second argument of the kernel.
    :param observations: List of observations. Size should match the size of the list of operators.
    It should be a list of tuples, [(obs locations, obs values)], where obs locations has m rows and d columns and
    values has m rows and 1 column. Here m is the number of points observed and d is the dimension of each point.
    :param op_system: The operator system, which gives access to the compiled kernel for each combination of ops.
    Minimum requirement for this is that it should function as a dictionary accepting tuples of combinations
    of operators, and returning functions which produce gram matrices.
    :param fun_args: array of scalar extra arguments which are to be passed to the functions from the operator
    system.
    :return: CollocationPosterior for the supplied operators and observations.
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

    LLbar = calc_LLbar(operators, operators_bar, observations, op_system, fun_args)
    LLbar_inv = np.linalg.inv(LLbar)

    # finally return the posterior
    return CollocationPosterior(operators, operators_bar, op_system, observations, LLbar_inv, fun_args)


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


def calc_side_matrices(operators, operators_bar, obs, test_points, op_cache, fun_args=None, outer_ops=None):
    logger.debug('Calculating side matrices for {} operators; test points of shape {}'.format(
        len(operators),
        test_points.shape)
    )
    obs_points = [p for p, _ in obs]
    Identity = ()
    if outer_ops is None:
        outer_ops = [[Identity], [Identity]]
    outer_ops_bar = outer_ops[1]
    outer_ops = outer_ops[0]

    row = []
    row_bar = []
    for outer_op, outer_op_bar in zip(outer_ops, outer_ops_bar):
        L, Lbar = ([], [])
        for op, op_bar, point in zip(operators, operators_bar, obs_points):
            f = op_cache[(op, outer_op_bar)]
            fbar = op_cache[(op_bar, outer_op)]
            logging.info('Applying {}, {} to points ({}) and test points ({})'
                         .format((op, outer_op_bar), (op_bar, outer_op), point.shape, test_points.shape))
            L.append(f(point, test_points, fun_args))
            Lbar.append(fbar(test_points, point, fun_args))
        L = np.concatenate(L)
        Lbar = np.concatenate(Lbar, axis=1)
        row.append(L)
        row_bar.append(Lbar)
    row = np.column_stack(row)
    row_bar = np.row_stack(row_bar)

    logger.debug('Returning shapes: row={}, row_bar={}'.format(row.shape, row_bar.shape))
    return row, row_bar


class CollocationPosterior(object):
    def __init__(self, operators, operators_bar, op_cache, obs, LLbar_inv, fun_args=None, my_ops=None):
        self.__operators = operators
        self.__operators_bar = operators_bar
        self.__op_cache = op_cache
        self.__obs = obs
        self.__LLbar_inv__ = LLbar_inv
        self.__fun_args = fun_args
        self.__outer_ops__ = my_ops

    @property
    def ops(self):
        return self.__operators

    @property
    def ops_bar(self):
        return self.__operators_bar

    def __call__(self, test_points):
        return self.posterior(test_points)

    def posterior(self, test_points):
        g = np.concatenate([val for _, val in self.__obs])
        mu_multiplier, Sigma = self.no_obs_posterior(test_points)
        return np.dot(mu_multiplier, g), Sigma

    def sample(self, test_points, samples=1):
        mu, cov = self.posterior(test_points)
        return np.random.multivariate_normal(mu.ravel(), cov, samples)

    def mean(self, test_points, g=None):
        if g is None:
            g = np.concatenate([val for _, val in self.__obs])
        L, Lbar = calc_side_matrices(
            self.__operators,
            self.__operators_bar,
            self.__obs,
            test_points,
            self.__op_cache,
            self.__fun_args,
            self.__outer_ops__)
        mu_multiplier = np.dot(Lbar, self.__LLbar_inv__)

        return np.dot(mu_multiplier, g)

    def no_obs_posterior(self, test_points):
        L, Lbar = calc_side_matrices(
            self.__operators,
            self.__operators_bar,
            self.__obs,
            test_points,
            self.__op_cache,
            self.__fun_args,
            self.__outer_ops__
        )

        mu_multiplier = np.dot(Lbar, self.__LLbar_inv__)
        if self.__outer_ops__ is None:
            k_eval = self.__op_cache[()]
            k_mat = k_eval(test_points, test_points, self.__fun_args)
        else:
            synthetic_obs = [(test_points, None) for _ in self.__outer_ops__[0]]
            k_mat = calc_LLbar(
                self.__outer_ops__[0],
                self.__outer_ops__[1],
                synthetic_obs,
                self.__op_cache,
                self.__fun_args
            )
        Sigma = k_mat - np.dot(mu_multiplier, L)

        return mu_multiplier, Sigma

    def diagonal_covariance(self, test_points):
        ret = np.empty((test_points.shape[0], 1))
        for i in xrange(test_points.shape[0]):
            _, cov = self.no_obs_posterior(test_points[i, :][None, :])
            ret[i, 0] = cov
        return ret

    def apply_operator(self, ops, ops_bar):
        if self.__outer_ops__ is not None:
            raise Exception('Application of an operator to a posterior which has already been operated upon is not supported!')
        return CollocationPosterior(self.__operators,
                                    self.__operators_bar,
                                    self.__op_cache,
                                    self.__obs,
                                    self.__LLbar_inv__,
                                    self.__fun_args,
                                    (ops, ops_bar),
                                    )