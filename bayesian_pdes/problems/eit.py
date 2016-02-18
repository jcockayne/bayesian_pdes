__author__ = 'benorn'
import bayesian_pdes as bpdes
import itertools
import numpy as np
from problem_helpers import get_function, canonical_AAbar, canonical_A


class FastEITMatrixComputer(object):
    def __init__(self, interior, boundary, eval_pts, kernel, symbols):
        # build application matrices
        int = {}
        bdy = {}
        int_bdy = {}
        int_eval = {}
        bdy_eval = {}

        for a,b in itertools.product(symbols[0], symbols[1]):
            # interiors
            int[(a, b)] = int[(b, a)] = get_function(kernel, (a, b), symbols, interior, interior)
            int[(a,a,b,b)] = int[(b,b,a,a)] = get_function(kernel, (a,a,b,b), symbols, interior, interior)
            int[(a,b,b)] = int[(b,b,a)] = get_function(kernel, (a,b,b), symbols, interior, interior)
            int[(a,a,b)] = int[(b,a,a)] = get_function(kernel, (a,a,b), symbols, interior, interior)

            # boundaries
            bdy[(a,b)] = bdy[(b,a)] = get_function(kernel, (a,b), symbols, boundary, boundary)

            # interior-boundary crossover
            int_bdy[(a,b)] = int_bdy[(b,a)] = get_function(kernel, (a,b), symbols, interior, boundary)
            int_bdy[(a,a,b)] = int_bdy[b,a,a] = get_function(kernel, (a,a,b), symbols, interior, boundary)

            # interior-eval crossover
            int_eval[(a)] = get_function(kernel, (a,), symbols, interior, eval_pts)
            int_eval[(a,a)] = get_function(kernel, (a,a), symbols, interior, eval_pts)

            # bdy-eval crossover
            bdy_eval[(a)] = get_function(kernel, (a,), symbols, boundary, eval_pts)

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

    # bdy
    def BBbar(self, kappa, kappa_x, kappa_y):
        bdy = self.__boundary
        interior = self.__interior
        matrices_bdy = self.__matrices_bdy
        x_1, x_2 = self.__symbols[0]
        y_1, y_2 = self.__symbols[1]

        kappa = kappa[interior.shape[0]:kappa.shape[0]].reshape((bdy.shape[0], 1))
        x = bdy[:,0].reshape((bdy.shape[0], 1))
        y = bdy[:,1].reshape((bdy.shape[0], 1))

        return np.exp(kappa).dot(np.exp(kappa.T)) * (x.dot(x.T) * matrices_bdy[(x_1, y_1)] + x.dot(y.T) * matrices_bdy[(x_1, y_2)] \
            + y.dot(x.T) * matrices_bdy[(x_2, y_1)] + y.dot(y.T) * matrices_bdy[(x_2, y_2)])


    def ABbar(self, kappa, kappa_x, kappa_y):
        bdy = self.__boundary
        interior = self.__interior
        matrices_int_bdy = self.__matrices_int_bdy
        x_1, x_2 = self.__symbols[0]
        y_1, y_2 = self.__symbols[1]

        x = bdy[:,0].reshape((bdy.shape[0], 1))
        y = bdy[:,1].reshape((bdy.shape[0], 1))
        kappa_int = kappa[:interior.shape[0]].reshape((interior.shape[0], 1))
        kappa_bdy = kappa[interior.shape[0]:kappa.shape[0]].reshape((bdy.shape[0], 1))
        kappa_x = kappa_x[:interior.shape[0]].reshape((interior.shape[0], 1))
        kappa_y = kappa_y[:interior.shape[0]].reshape((interior.shape[0], 1))

        term_a = kappa_x.dot(x.T) * matrices_int_bdy[(x_1, y_1)] \
            + kappa_x.dot(y.T) * matrices_int_bdy[(x_1, y_2)] \
            + kappa_y.dot(x.T) * matrices_int_bdy[(x_2, y_1)] \
            + kappa_y.dot(y.T) * matrices_int_bdy[(x_2, y_2)]
        term_b = x.T * matrices_int_bdy[(x_1, x_1, y_1)] \
            + y.T * matrices_int_bdy[(x_1, x_1, y_2)] \
            + x.T * matrices_int_bdy[(x_2, x_2, y_1)] \
            + y.T * matrices_int_bdy[(x_2, x_2, y_2)]

        return np.exp(kappa_int).dot(np.exp(kappa_bdy.T)) * (term_a + term_b)

    def A(self, kappa, kappa_x, kappa_y):
        interior = self.__interior
        matrices_int_eval = self.__matrices_int_eval
        return canonical_A(interior, matrices_int_eval, self.__symbols, kappa, kappa_x, kappa_y)

    def B(self, kappa, kappa_x, kappa_y):
        bdy = self.__boundary
        x_1, x_2 = self.__symbols[0]
        matrices_bdy_eval = self.__matrices_bdy_eval
        shape = (bdy.shape[0], 1)

        kappa = kappa[-bdy.shape[0]:].reshape(shape)

        x = bdy[:,0].reshape((bdy.shape[0], 1))
        y = bdy[:,1].reshape((bdy.shape[0], 1))

        return np.exp(kappa) * (x * matrices_bdy_eval[(x_1)] + y * matrices_bdy_eval[(x_2)])