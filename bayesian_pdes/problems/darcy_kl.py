from __future__ import print_function
import logging
from util import NamedLambda
import bayesian_pdes as bpdes

logger = logging.getLogger(__name__)

try:
    from autograd import numpy as np
    from autograd.scipy import stats
except Exception as ex:
    logger.warning("Failed to import autograd! Gradients will not be available.")
    import numpy as np
    from scipy import stats


def cfun(x, theta):
    log_c = 0. \
            + theta[0] * np.cos(2 * np.pi * x[:, 0]) \
            + theta[1] * np.cos(2 * np.pi * x[:, 1]) \
            + theta[2] * np.cos(2 * np.pi * (x[:, 0] + x[:, 1])) \
            + theta[3] * np.cos(4 * np.pi * x[:, 0]) \
            + theta[4] * np.cos(4 * np.pi * x[:, 1])
    return np.exp(log_c)


def log_c_x_fun(x, theta):
    return - theta[0] * 2 * np.pi * np.sin(2 * np.pi * x[:, 0]) \
           - theta[2] * 2 * np.pi * np.sin(2 * np.pi * (x[:, 0] + x[:, 1])) \
           - theta[3] * 4 * np.pi * np.sin(4 * np.pi * (x[:, 0]))


def log_c_y_fun(x, theta):
    return - theta[1] * 2 * np.pi * np.sin(2 * np.pi * x[:, 1]) \
           - theta[2] * 2 * np.pi * np.sin(2 * np.pi * (x[:, 0] + x[:, 1])) \
           - theta[4] * 4 * np.pi * np.sin(4 * np.pi * (x[:, 1]))


class DarcyKLFactory(object):
    def __init__(self, kernel, symbols, compile_extra_args=None, verbosity=0):
        if compile_extra_args is None:
            compile_extra_args = {}

        self.__verbosity__ = verbosity
        symbols, symbols_bar, extra_symbols = symbols
        x_1, x_2 = symbols
        y_1, y_2 = symbols_bar

        # Operators in the expanded version of the problem
        A_1 = NamedLambda(lambda k: k.diff(x_1), 'A_1')
        A_2 = NamedLambda(lambda k: k.diff(x_2), 'A_2')
        A_3 = NamedLambda(lambda k: k.diff(x_1, x_1) + k.diff(x_2, x_2), 'A_3')
        A_1_bar = NamedLambda(lambda k: k.diff(y_1), 'A_1_bar')
        A_2_bar = NamedLambda(lambda k: k.diff(y_2), 'A_2_bar')
        A_3_bar = NamedLambda(lambda k: k.diff(y_1, y_1) + k.diff(y_2, y_2), 'A_3_bar')

        # Transformed operators
        A_t = 'A_t'
        A_bar_t = 'A_bar_t'

        def Identity(k):
            return k

        # Boundary operators
        B_1 = NamedLambda(Identity, 'B_1')
        B_1_bar = NamedLambda(Identity, 'B_1_bar')
        B_2 = NamedLambda(lambda k: k.diff(x_1), 'B_2')
        B_2_bar = NamedLambda(lambda k: k.diff(y_1), 'B_2_bar')

        ops = [A_t, B_1, B_2]
        ops_bar = [A_bar_t, B_1_bar, B_2_bar]

        ops_base = [A_1, A_2, A_3, B_1, B_2]
        ops_bar_base = [A_1_bar, A_2_bar, A_3_bar, B_1_bar, B_2_bar]

        op_cache_base = bpdes.operator_compilation.sympy_gram.compile_sympy(
            ops_base,
            ops_bar_base,
            kernel,
            [symbols, symbols_bar, extra_symbols],
            **compile_extra_args
        )

        self.__ops__ = ops
        self.__ops_bar__ = ops_bar
        self.__op_cache_base__ = op_cache_base
        self.__caching_op_system__ = bpdes.operator_compilation.CachingOpCache(op_cache_base)
        self.__gradient_function__ = None

    def get_operator_system(self, theta, interior_points, use_cache=True):
        op_system = self.__caching_op_system__ if use_cache else self.__op_cache_base__
        return TransformedOpCache(self.__ops__, self.__ops_bar__, op_system, theta, interior_points,
                                  self.__verbosity__)

    def clear_cache(self):
        self.__caching_op_system__.clear()

    def log_likelihood(self, theta, observations, truth, length_scale, likelihood_sigma, use_cache=True):
        ls = np.array([length_scale])
        interior_points = observations[0][0]
        oc = self.get_operator_system(theta, interior_points, use_cache)
        posterior = bpdes.collocate(oc.operators, oc.operators_bar, observations, oc, fun_args=ls)

        true_x, true_u = truth

        mu, cov = posterior(true_x)
        mu = mu.reshape((len(mu), 1))

        cov_with_error = cov + likelihood_sigma**2*np.eye(cov.shape[0])
        return stats.multivariate_normal.logpdf(mu.ravel(), true_u.ravel(), cov_with_error)

    def grad_log_likelihood(self, *args, **kwargs):
        if self.__gradient_function__ is None:
            try:
                import autograd
            except ImportError:
                raise Exception("Unable to import autograd; gradient information is unavailable.")
            self.__gradient_function__ = autograd.grad(self.log_likelihood)
        return self.__gradient_function__(*args, **kwargs)


class TransformedOpCache(object):
    def __init__(self, operators, operators_bar, base_op_system, theta, interior_points, verbosity=0):
        self.__operators__ = operators
        self.__operators_bar__ = operators_bar
        self.__base_op_system__ = base_op_system
        self.__theta__ = theta
        self.__verbosity__ = verbosity
        self.__interior__ = interior_points

    @property
    def operators(self):
        return self.__operators__

    @property
    def operators_bar(self):
        return self.__operators_bar__

    def __getitem__(self, item):
        return self.do_transform(item)

    def do_transform(self, item):
        # set up required variables
        A_t, B_1, B_2 = self.__operators__
        A_bar_t, B_1_bar, B_2_bar = self.__operators_bar__

        A_1, A_2, A_3, B_1, B_2 = self.__base_op_system__.operators
        A_1_bar, A_2_bar, A_3_bar, B_1_bar, B_2_bar = self.__base_op_system__.operators_bar

        op_cache = self.__base_op_system__
        theta = self.__theta__
        interior = self.__interior__

        def printer(*args):
            if self.__verbosity__ > 0:
                print(*args)

        c = cfun(interior, theta)[:, None]
        grad_log_c_x = log_c_x_fun(interior, theta)[:, None]
        grad_log_c_y = log_c_y_fun(interior, theta)[:, None]
        all_things = [()]

        # first explode out the objects required
        for i in item:
            if i == A_t:
                all_things = sum([[a + (A_1,), a + (A_2,), a + (A_3,)] for a in all_things], [])
            elif i == A_bar_t:
                all_things = sum([[a + (A_1_bar,), a + (A_2_bar,), a + (A_3_bar,)] for a in all_things], [])
            else:
                all_things = [a + (i,) for a in all_things]
        printer('{} -> {}'.format(item, all_things))

        def __calc_result__(x, y, fun_args):
            result = 0
            for my_item in all_things:
                try:
                    function = op_cache[my_item]
                except Exception as ex:
                    printer('Failed to get {}'.format(my_item))
                    raise ex
                new_mat = function(x, y, fun_args)
                if A_1 in my_item:
                    printer('A_1')
                    new_mat = np.repeat(grad_log_c_x * c, y.shape[0], 1) * new_mat
                if A_2 in my_item:
                    printer('A_2')
                    new_mat = np.repeat(grad_log_c_y * c, y.shape[0], 1) * new_mat
                elif A_3 in my_item:
                    printer('A_3')
                    new_mat = np.repeat(c, y.shape[0], 1) * new_mat
                if A_1_bar in my_item:
                    printer('A_1_bar')
                    new_mat = np.repeat(grad_log_c_x.T * c.T, x.shape[0], 0) * new_mat
                if A_2_bar in my_item:
                    printer('A_2_bar')
                    new_mat = np.repeat(grad_log_c_y.T * c.T, x.shape[0], 0) * new_mat
                elif A_3_bar in my_item:
                    printer('A_3_bar')
                    new_mat = np.repeat(c.T, x.shape[0], 0) * new_mat
                # autograd doesn't like +=
                result = result + new_mat
            return result

        return __calc_result__
