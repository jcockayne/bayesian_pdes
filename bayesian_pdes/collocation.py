try:
    from autograd import numpy as np
except:
    print("Autograd not available; using standard numpy.")
    import numpy as np

import logging
import time
from util import linalg

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
    for o in observations:
        locs = o[0]
        if len(o) > 1:
            vals = o[1]
        else:
            vals = None
        if vals is None: continue
        err = " (Loc has shape {}, vals have shape {})".format(locs.shape, vals.shape)
        if locs.shape[0] != vals.shape[0]:
            raise Exception("Number of obs not consistent with location of obs " + err)
        if len(locs.shape) != 2:
            raise Exception("Obs locations must be two-dimensional " + err)

        if len(o) > 2:
            cov = o[2]
            if cov.shape != (locs.shape[0], locs.shape[0]):
                raise Exception('Size of cov {} not consistent with locations {}'.format(cov.shape, locs.shape))

    if len(operators) != len(observations):
        raise Exception('Number of obs not consistent with number of operators ({} observations but {} operators)'
                        .format(len(observations), len(operators)))

    LLbar = calc_LLbar(operators, operators_bar, observations, op_system, fun_args)
    LLbar_inv = np.linalg.inv(LLbar)

    # finally return the posterior
    return CollocationPosterior(operators, operators_bar, op_system, observations, LLbar_inv, fun_args)


def compute_operator_matrix(operators, operators_bar, points, points_bar, op_system, fun_args=None):
    """
    Compute the matrix formed by applying all combinations of operators, operators_bar to the kernel described in
    the operator system, when applied to the corresponding supplied points.
    :param operators: operators to apply
    :param operators_bar: barred operators to apply
    :param points: points corresponding to operators
    :param points_bar: points corresponding to barred operators
    :param op_system: the operator system
    :param fun_args: any additional arguments to functions in the operator system
    :return: the required matrix.
    """
    if type(operators) is not list:
        operators = [operators]
    if type(operators_bar) is not list:
        operators_bar = [operators_bar]
    if type(points) is not list:
        points = [points for _ in operators]
    if type(points_bar) is not list:
        points_bar = [points_bar for _ in operators_bar]
    rows = []
    assert len(operators) == len(points), \
        'Inconsistent number of operators {} compared with points {}'.format(len(operators), len(points))
    assert len(operators_bar) == len(points_bar), \
        'Inconsistent number of operators {} compared with points {}'.format(len(operators_bar), len(points_bar))
    t = time.time()
    for op, p in zip(operators, points):
        row = []
        for op_bar, p_bar in zip(operators_bar, points_bar):
            fun_op = op_system[(op, op_bar)]
            msg = 'Calling {} with arg shapes x={}, y={}'.format((op, op_bar), p.shape, p_bar.shape)
            if fun_args is not None:
                msg += ', args={}'.format(fun_args.shape)
            logger.debug(msg)
            applied = fun_op(p, p_bar, fun_args)
            row.append(applied)
        rows.append(np.column_stack(row))
    logger.debug('compute_operator_matrix for {}, {} took {}s'.format(operators, operators_bar, time.time() - t))
    return np.row_stack(rows)


def calc_LLbar(operators, operators_bar, observations, op_cache, fun_args):
    points = [p[0] for p in observations]

    op_mat = compute_operator_matrix(operators, operators_bar, points, points, op_cache, fun_args)
    if not any(len(o) == 3 for o in observations):
        return op_mat

    # adjust for covariance of observations
    covs = []
    for o in observations:
        points = o[0]
        num_points = points.shape[0]
        if len(o) == 3:
            covs.append(o[2])
        else:
            covs.append(np.zeros((num_points, num_points)))
    adj = linalg.block_diag(covs)
    return op_mat + adj


def calc_side_matrices(operators, operators_bar, obs, test_points, op_cache, outer_ops=None, outer_ops_bar=None, fun_args=None):
    if outer_ops is None:
        outer_ops = ()
    if outer_ops_bar is None:
        outer_ops_bar = ()
    points = [p[0] for p in obs]
    L = compute_operator_matrix(operators, outer_ops_bar, points, test_points, op_cache, fun_args)
    Lbar = compute_operator_matrix(outer_ops, operators_bar, test_points, points, op_cache, fun_args)
    return L, Lbar


class CollocationPosterior(object):
    def __init__(self, operators, operators_bar, op_cache, obs, LLbar_inv, fun_args=None, my_ops=None, my_ops_bar=None, prior=None):
        self.__operators__ = operators
        self.__operators_bar__ = operators_bar
        self.__op_system__ = op_cache
        self.__obs__ = obs
        self.__LLbar_inv__ = LLbar_inv
        self.__fun_args__ = fun_args
        self.__outer_ops__ = [()] if my_ops is None else my_ops
        self.__outer_ops_bar__ = [()] if my_ops_bar is None else my_ops_bar
        self.__prior__ = prior

    @property
    def ops(self):
        return self.__operators__

    @property
    def ops_bar(self):
        return self.__operators_bar__

    def __call__(self, test_points):
        return self.posterior(test_points)

    def posterior(self, test_points):
        g = np.concatenate([o[1] for o in self.__obs__])
        mu_multiplier, Sigma = self.no_obs_posterior(test_points)
        return self.__adjust_mean__(test_points, g, mu_multiplier), Sigma

    def kern(self, x, y, fun_args):
        assert all([len(o) < 3 for o in self.__obs__]), "Currently this method doesn't support noisy obs properly!"
        obs_points = [o[0] for o in self.__obs__]
        L = compute_operator_matrix(self.__operators__, self.__outer_ops_bar__, obs_points, y, self.__op_system__, fun_args)
        Lbar = compute_operator_matrix(self.__outer_ops__, self.__operators_bar__, x, obs_points, self.__op_system__, fun_args)
        k_mat = compute_operator_matrix(self.__outer_ops__, self.__outer_ops_bar__, x, y, self.__op_system__, fun_args)

        return k_mat - np.dot(Lbar, np.dot(self.__LLbar_inv__, L))

    def sample(self, test_points, samples=1):
        mu, cov = self.posterior(test_points)
        return np.random.multivariate_normal(mu.ravel(), cov, samples).T

    def mean(self, test_points, g=None):
        if g is None:
            g = np.concatenate([v[1] for v in self.__obs__])

        obs_points = [v[0] for v in self.__obs__]
        Lbar = compute_operator_matrix(self.__outer_ops__,
                                       self.__operators_bar__,
                                       test_points,
                                       obs_points,
                                       self.__op_system__,
                                       self.__fun_args__)
        mu_multiplier = np.dot(Lbar, self.__LLbar_inv__)

        return self.__adjust_mean__(test_points, g, mu_multiplier)

    def __adjust_mean__(self, test_points, g, multiplier):
        prior = self.__prior__

        if prior is not None:
            prior_mean_obs = [calc_mean(prior, o, p[0]) for o, p in zip(self.__operators__, self.__obs__)]
            prior_mean_obs = np.row_stack(prior_mean_obs)
            prior_mean_test = calc_mean(prior, self.__outer_ops__, test_points)
        else:
            prior_mean_test = 0
            prior_mean_obs = 0
        return prior_mean_test + np.dot(multiplier, (g - prior_mean_obs))

    def no_obs_posterior(self, test_points):

        points = [p[0] for p in self.__obs__]
        L = compute_operator_matrix(self.__operators__, self.__outer_ops_bar__, points, test_points, self.__op_system__, self.__fun_args__)
        Lbar = compute_operator_matrix(self.__outer_ops__, self.__operators_bar__, test_points, points, self.__op_system__, self.__fun_args__)

        mu_multiplier = np.dot(Lbar, self.__LLbar_inv__)
        k_mat = compute_operator_matrix(
            self.__outer_ops__,
            self.__outer_ops_bar__,
            test_points,
            test_points,
            self.__op_system__,
            self.__fun_args__)

        logger.debug('Shapes L:{} Lbar:{} LLbar_inv: {} kmat:{}'.format(L.shape, Lbar.shape, self.__LLbar_inv__.shape, k_mat.shape))

        right_term = np.dot(mu_multiplier, L)
        Sigma = k_mat - right_term

        return mu_multiplier, Sigma

    def diagonal_covariance(self, test_points):
        ret = np.empty((test_points.shape[0], 1))
        for i in xrange(test_points.shape[0]):
            _, cov = self.no_obs_posterior(test_points[i, :][None, :])
            ret[i, 0] = cov
        return ret

    def apply_operators(self, ops, ops_bar):
        if self.__outer_ops__ != [()]:
            raise Exception('Application of an operator to a posterior which has already been operated upon is not supported!')
        return CollocationPosterior(self.__operators__,
                                    self.__operators_bar__,
                                    self.__op_system__,
                                    self.__obs__,
                                    self.__LLbar_inv__,
                                    self.__fun_args__,
                                    ops,
                                    ops_bar,
                                    self.__prior__
                                    )


def calc_mean(prior, op, points):
            operated = prior.apply_operators(op, ())
            return operated.mean(points)