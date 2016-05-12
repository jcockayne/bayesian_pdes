from __future__ import print_function
import bayesian_pdes
import numpy as np

class LaplacianInverseProblem(object):
    def __init__(self, base_op_cache, theta, verbosity=0, natural_kernel=False):
        self.verbosity = verbosity
        self.theta = theta
        self.A = 'A'
        self.A_bar = 'A_bar'
        self.operators = [self.A]
        self.operators_bar = [self.A_bar]
        self.natural_kernel = natural_kernel

        self.__base_system__ = base_op_cache
        if len(base_op_cache.operators) == 2:
            self.operators.append(base_op_cache.operators[1])
            self.operators_bar.append(base_op_cache.operators_bar[1])

    def __getitem__(self, item):
        return self.do_transform(item)

    def do_transform(self, item):
        if type(item) is not tuple:
            item = (item,)

        def printer(*args):
            if self.verbosity > 0:
                print(*args)

        printer('Getting {}'.format(item))

        new = ()
        for i in item:
            if i == self.A:
                new += (self.__base_system__.operators[0],)
            elif i == self.A_bar:
                new += (self.__base_system__.operators_bar[0],)
            else:
                new += (i,)

        def __calc_result__(x, y, fun_args=None):
            if fun_args is None:
                fun_args = np.array([])
            res = self.__base_system__[new]
            mat = res(x, y, fun_args)

            if self.A in item:
                mat = mat * self.theta
            if self.A_bar in item:
                mat = mat * self.theta
            if self.natural_kernel:
                mat = mat / (self.theta**2)
            return mat
        return __calc_result__


class LaplacianInverseProblemFactory(object):
    def __init__(self, base_system, verbosity=0, natural_kernel=False):
        self.base_system = bayesian_pdes.operator_compilation.CachingOpCache(base_system)
        self.verbosity = verbosity
        self.natural_kernel = natural_kernel

    def get_operator_system(self, theta):
        return LaplacianInverseProblem(self.base_system, theta, self.verbosity, self.natural_kernel)