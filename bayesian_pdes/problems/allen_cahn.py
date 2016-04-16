from __future__ import print_function
from util import NamedLambda
import sympy as sp
import bayesian_pdes


class AllenCahnOpSystem(object):
    def __init__(self, base_op_system, delta, verbosity=0):
        self.__base_system__ = base_op_system
        self.__delta__ = delta
        self.__verbosity__ = verbosity

        A_t = 'A_t'
        A_bar_t = 'A_bar_t'

        B = self.__base_system__.operators[2]
        B_bar = self.__base_system__.operators_bar[2]

        A_2 = self.__base_system__.operators[3]
        A_2_bar = self.__base_system__.operators_bar[3]

        self.operators = [A_t, A_2, B]
        self.operators_bar = [A_bar_t, A_2_bar, B_bar]

    def __getitem__(self, item):
        return self.do_transform(item, self.__base_system__, self.__delta__, self.__verbosity__)

    def do_transform(self, item, op_cache, delta, verbosity):
        if type(item) is not tuple:
            item = (item,)
        def printer(*args):
            if verbosity > 0:
                print(*args)

        printer('Getting {}'.format(item))

        A_t = self.operators[0]
        A_bar_t = self.operators_bar[0]

        A_1 = self.__base_system__.operators[0]
        A_2 = self.__base_system__.operators[1]

        A_1_bar = self.__base_system__.operators_bar[0]
        A_2_bar = self.__base_system__.operators_bar[1]

        all_things = [()]
        # first explode out the objects required
        for i in item:
            if i == A_t:
                printer('Detected A_t; expanding operators')
                all_things = sum([[a + (A_1,), a + (A_2,)] for a in all_things], [])
            elif i == A_bar_t:
                printer('Detected A_bar_t; expanding operators')
                all_things = sum([[a + (A_1_bar,), a + (A_2_bar,)] for a in all_things], [])
            else:
                all_things = [a + (i,) for a in all_things]
        printer('Operators: {} -> {}'.format(item, all_things))

        def __calc_result(x,y,fun_args=None):
            printer('x shape {} ({}), y shape {} ({})'.format(x.shape, x.dtype, y.shape, y.dtype))
            if fun_args is None:
                fun_args = []
            result = 0
            for item in all_things:
                try:
                    function = op_cache[item]
                except Exception as ex:
                    printer('Failed to get {}'.format(item))
                    raise ex
                new_mat = function(x, y, fun_args)
                if A_1 in item:
                    printer('Transforming A_1')
                    new_mat = -delta * new_mat
                if A_2 in item:
                    printer('Transforming A_2')
                    new_mat = -1./delta * new_mat
                if A_1_bar in item:
                    printer('Transforming A_1_bar')
                    new_mat = -delta * new_mat
                if A_2_bar in item:
                    printer('Transforming A_2_bar')
                    new_mat = -1./delta * new_mat
                printer('Matrix shape: {}'.format(new_mat.shape))
                result += new_mat
            return result
        return __calc_result


class AllenCahnFactory(object):
    def __init__(self):
        length_scale = sp.Symbol('sigma')
        x_1, x_2, y_1, y_2 = sp.symbols('x_1 x_2 y_1 y_2')
        k_sqexp = sp.exp(-((x_1 - y_1)**2 + (x_2 - y_2)**2) / (2*length_scale**2))
        A_1 = NamedLambda(lambda k: k.diff(x_1, x_1) + k.diff(x_2, x_2), 'A_1')
        A_1_bar = NamedLambda(lambda k: k.diff(y_1, y_1) + k.diff(y_2, y_2), 'A_1_bar')
        A_2 = NamedLambda(lambda k: k, 'A_2')
        A_2_bar = NamedLambda(lambda k: k, 'A_2_bar')
        B_1 = NamedLambda(lambda k: k, 'B_1')
        B_1_bar = NamedLambda(lambda k: k, 'B_1_bar')
        Identity = NamedLambda(lambda k: k, 'Identity')

        op_cache_base = bayesian_pdes.operator_compilation.compile_sympy([A_1, A_2, B_1, Identity],
                                                                         [A_1_bar, A_2_bar, B_1_bar, Identity],
                                                                         k_sqexp,
                                                                         [[x_1, x_2], [y_1, y_2], length_scale],
                                                                        )
        arg_caching = bayesian_pdes.operator_compilation.CachingOpCache(op_cache_base)
        self.__base_operator_system__ = arg_caching

    def get_operator_system(self, delta, verbosity=0):
        return AllenCahnOpSystem(self.__base_operator_system__, delta, verbosity)