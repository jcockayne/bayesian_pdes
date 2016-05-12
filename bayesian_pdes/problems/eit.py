from __future__ import print_function
import bayesian_pdes
try:
    from autograd import numpy as np
except Exception as ex:
    import numpy as np
from util import NamedLambda


class EITFactory(object):

    def __init__(self, kernel, symbols, symbols_bar, extra_symbols, compile_mode=None, verbosity=0):
        self.__verbosity__ = verbosity
        x_1, x_2 = symbols
        y_1, y_2 = symbols_bar

        # Operators in the expanded version of the problem
        A_1 = NamedLambda(lambda k: k.diff(x_1), 'A_1')
        A_2 = NamedLambda(lambda k: k.diff(x_2), 'A_2')
        A_3 = NamedLambda(lambda k: k.diff(x_1, x_1) + k.diff(x_2, x_2), 'A_3')

        A_1_bar = NamedLambda(lambda k: k.diff(y_1), 'A_1_bar')
        A_2_bar = NamedLambda(lambda k: k.diff(y_2), 'A_2_bar')
        A_3_bar = NamedLambda(lambda k: k.diff(y_1, y_1) + k.diff(y_2, y_2), 'A_3_bar')

        # Boundary operators
        B = NamedLambda(lambda k: (k.diff(x_1)*x_1 + k.diff(x_2)*x_2), 'B')
        B_bar = NamedLambda(lambda k: (k.diff(y_1)*y_1 + k.diff(y_2)*y_2), 'B_bar')

        ops_base = [A_1, A_2, A_3, B]
        ops_bar_base = [A_1_bar, A_2_bar, A_3_bar, B_bar]

        # Labels for the transformed operators
        A_t = 'A_t'
        A_bar_t = 'A_bar_t'
        B_t = 'B_t'
        B_bar_t = 'B_bar_t'

        self.__ops__ = [A_t, B_t]
        self.__ops_bar__ = [A_bar_t, B_bar_t]

        symbols = [symbols, symbols_bar]
        if extra_symbols is not None:
            symbols.append(extra_symbols)
        op_cache_base = bayesian_pdes.operator_compilation.compile_sympy(
            ops_base,
            ops_bar_base,
            kernel,
            symbols,
            mode=compile_mode
        )
        self.__base_op_cache__ = op_cache_base
        self.__caching_op_cache__ = bayesian_pdes.operator_compilation.CachingOpCache(op_cache_base)

    def clear_cache(self):
        self.__caching_op_cache__.clear()

    def get_operator_system(self, kappa_int, kappa_bdy, grad_kappa_x, grad_kappa_y, use_cache=False):
        op_cache = self.__caching_op_cache__ if use_cache else self.__base_op_cache__
        return EITOperatorSystem(self.__ops__,
                                 self.__ops_bar__,
                                 op_cache,
                                 kappa_int,
                                 kappa_bdy,
                                 grad_kappa_x,
                                 grad_kappa_y,
                                 self.__verbosity__)


class EITOperatorSystem(object):
    def __init__(self, operators, operators_bar, op_cache, kappa_int, kappa_bdy, grad_kappa_x, grad_kappa_y, verbosity=0):
        self.operators = operators
        self.operators_bar = operators_bar
        self.__op_cache__ = op_cache
        self.__kappa_int__ = kappa_int
        self.__kappa_bdy__ = kappa_bdy
        self.__grad_kappa_x__ = grad_kappa_x
        self.__grad_kappa_y__ = grad_kappa_y
        self.__verbosity__ = verbosity

    def __getitem__(self, item):
        return self.do_transform(item)

    def do_transform(self, item):
        if type(item) is not tuple:
            item = (item,)
        def printer(*args):
            if self.__verbosity__ > 0:
                print(*args)

        printer('Attempting to get {}'.format(item))

        A_t, B_t = self.operators
        A_bar_t, B_bar_t = self.operators_bar

        A_1, A_2, A_3, B = self.__op_cache__.operators
        A_1_bar, A_2_bar, A_3_bar, B_bar = self.__op_cache__.operators_bar

        exp_kappa_int = np.exp(self.__kappa_int__).reshape((len(self.__kappa_int__), 1))
        exp_kappa_bdy = np.exp(self.__kappa_bdy__).reshape((len(self.__kappa_bdy__), 1))
        grad_kappa_x = self.__grad_kappa_x__.reshape((len(self.__grad_kappa_x__), 1))
        grad_kappa_y = self.__grad_kappa_y__.reshape((len(self.__grad_kappa_y__), 1))

        all_things = [()]

        printer(exp_kappa_int.shape, exp_kappa_bdy.shape, grad_kappa_x.shape, grad_kappa_y.shape)

        # first explode out the objects required
        for i in item:
            if i == A_t:
                all_things = sum([[a + (A_1,), a + (A_2,), a + (A_3,)] for a in all_things], [])
            elif i == A_bar_t:
                all_things = sum([[a + (A_1_bar,), a + (A_2_bar,), a + (A_3_bar,)] for a in all_things], [])
            elif i == B_t:
                all_things = [a + (B,) for a in all_things]
            elif i == B_bar_t:
                all_things = [a + (B_bar,) for a in all_things]
            else:
                all_things = [a + (i,) for a in all_things]
        printer('Mapped {} to {}'.format(item, all_things))

        def __ret(x, y, fun_args=None):
            return self.calc_result(
                x,
                y,
                fun_args,
                all_things,
                exp_kappa_int,
                exp_kappa_bdy,
                grad_kappa_x,
                grad_kappa_y
            )
        return __ret

    def calc_result(self, x, y, fun_args, all_things, exp_kappa_int, exp_kappa_bdy, grad_kappa_x, grad_kappa_y):
        result = 0

        A_1, A_2, A_3, B = self.__op_cache__.operators
        A_1_bar, A_2_bar, A_3_bar, B_bar = self.__op_cache__.operators_bar

        def printer(*args):
            if self.__verbosity__ > 0:
                print(*args)

        for item in all_things:
            try:
                function = self.__op_cache__[item]
            except Exception as ex:
                printer('Failed to get {}'.format(item))
                raise ex
            new_mat = function(x, y, fun_args)



            # unbarred
            if A_1 in item:
                printer('Transforming A_1')
                multiplier = np.repeat(grad_kappa_x*exp_kappa_int, y.shape[0], 1)
                new_mat = multiplier * new_mat
            elif A_2 in item:
                printer('Transforming A_2')
                multiplier = np.repeat(grad_kappa_y*exp_kappa_int, y.shape[0], 1)
                new_mat = multiplier * new_mat
            elif A_3 in item:
                printer('Transforming A_3')
                multiplier = np.repeat(exp_kappa_int, y.shape[0], 1)
                new_mat = multiplier * new_mat

            # barred
            if A_1_bar in item:
                printer('Transforming A_1_bar')
                new_mat = np.repeat(grad_kappa_x.T*exp_kappa_int.T,x.shape[0],0) * new_mat
            elif A_2_bar in item:
                printer('Transforming A_2_bar')
                new_mat = np.repeat(grad_kappa_y.T*exp_kappa_int.T,x.shape[0],0) * new_mat
            elif A_3_bar in item:
                printer('Transforming A_3_bar')
                new_mat = np.repeat(exp_kappa_int.T,x.shape[0],0) * new_mat

            # boundary
            if B in item:
                printer('Transforming B')
                new_mat = np.repeat(exp_kappa_bdy, y.shape[0], 1) * new_mat
            if B_bar in item:
                printer('Transforming B_bar')
                new_mat = np.repeat(exp_kappa_bdy.T, x.shape[0], 0) * new_mat
            result += new_mat
        return result
