__author__ = 'benorn'
import itertools
from problem_helpers import get_function, canonical_A, canonical_AAbar
class FastCanonicalDirichletMatrixComputer(object):
    def __init__(self, interior, boundary, eval_pts, kernel, symbols):
        # build application matrices
        int = {}
        bdy = {}
        int_bdy = {}
        int_eval = {}
        bdy_eval = {}

        for a,b in itertools.product(symbols[0], symbols[1]):
            # interiors
            int[(a,b)] = int[(b, a)] = get_function(kernel, (a, b), symbols, interior, interior)
            int[(a,a,b,b)] = int[(b,b,a,a)] = get_function(kernel, (a,a,b,b), symbols, interior, interior)
            int[(a,b,b)] = int[(b,b,a)] = get_function(kernel, (a,b,b), symbols, interior, interior)
            int[(a,a,b)] = int[(b,a,a)] = get_function(kernel, (a,a,b), symbols, interior, interior)

            # boundaries
            bdy[()] = get_function(kernel, (), symbols, boundary, boundary)
            bdy[(a)] = get_function(kernel, (a,), symbols, boundary, boundary)
            bdy[(a,b)] = bdy[(b,a)] = get_function(kernel, (a,b), symbols, boundary, boundary)

            # interior-boundary crossover
            int_bdy[(a,b)] = int_bdy[(b,a)] = get_function(kernel, (a,b), symbols, interior, boundary)
            int_bdy[(a,a,b)] = int_bdy[(b,a,a)] = get_function(kernel, (a,a,b), symbols, interior, boundary)

            # interior-eval crossover
            int_eval[(a)] = get_function(kernel, (a,), symbols, interior, eval_pts)
            int_eval[(a,a)] = get_function(kernel, (a,a), symbols, interior, eval_pts)

            # bdy-eval crossover
            bdy_eval[()] = get_function(kernel, (), symbols, boundary, eval_pts)
            bdy_eval[(a)] = get_function(kernel, (a,), symbols, boundary, boundary)

        self.__interior = interior
        self.__boundary = boundary
        self.__matrices_int = int
        self.__matrices_bdy = bdy
        self.__matrices_int_bdy = int_bdy
        self.__matrices_int_eval = int_eval
        self.__matrices_bdy_eval = bdy_eval
        self.__symbols = symbols

    def AAbar(self, kappa, kappa_x, kappa_y):
        interior = self.__interior
        matrices_int = self.__matrices_int
        return canonical_AAbar(interior, matrices_int, self.__symbols, kappa, kappa_x, kappa_y)

    def A(self, kappa, kappa_x, kappa_y):
        return canonical_A(self.__interior, self.__matrices_int_eval, self.__symbols, kappa, kappa_x, kappa_y)

    def B_1(self, kappa, kappa_x, kappa_y):
        return self.__matrices_bdy_eval[()]

    def B_1B_1bar(self, kappa, kappa_x, kappa_y):
        return self.__matrices_bdy[()]

    def AB_1bar(self, kappa, kappa_x, kappa_y):
        return canonical_A(self.__interior, self.__matrices_int_bdy, self.__symbols, kappa, kappa_x, kappa_y)

    # B_2 is the operator which gives first derivative wrt. x_1
    def B_2(self, kappa, kappa_x, kappa_y):
        x_1, x_2 = self.__symbols[0]
        return self.__matrices_bdy_eval[(x_1)]

    def B_2B_2bar(self, kappa, kappa_x, kappa_y):
        x_1, x_2 = self.__symbols[0]
        y_1, y_2 = self.__symbols[1]
        return self.__matrices_bdy[(x_1, y_1)]

    def B_1B_2bar(self, kappa, kappa_x, kappa_y):
        x_1, x_2 = self.__symbols[0]
        return self.__matrices_bdy[(x_1)].T

    def AB_2bar(self, kappa, kappa_x, kappa_y):
        pass #todo