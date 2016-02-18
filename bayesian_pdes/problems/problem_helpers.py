__author__ = 'benorn'
import bayesian_pdes as bpdes
import numpy as np


def get_function(kernel, diff_symbols, fun_symbols, arg1, arg2):
    diffed = kernel.diff(*diff_symbols)
    fun = bpdes.sympy_helpers.sympy_function(diffed, fun_symbols)
    applied = bpdes.pairwise.apply(fun, arg1, arg2)
    return applied

def canonical_AAbar(interior, matrices, symbols, kappa, kappa_x, kappa_y):
    x_1, x_2 = symbols[0]
    y_1, y_2 = symbols[1]

    shape = (interior.shape[0], 1)
    kappa = kappa[:interior.shape[0]].reshape(shape)
    kappa_x = kappa_x[:interior.shape[0]].reshape(shape)
    kappa_y = kappa_y[:interior.shape[0]].reshape(shape)

    term_a = kappa_x.dot(kappa_x.T) * matrices[(x_1, y_1)]\
        + kappa_x.dot(kappa_y.T) * matrices[(x_1, y_2)]\
        + kappa_y.dot(kappa_x.T) * matrices[(x_2, y_1)]\
        + kappa_y.dot(kappa_y.T) * matrices[(x_2, y_2)]

    term_b = kappa_x * (matrices[(x_1, y_1, y_1)] + matrices[(x_1, y_2, y_2)]) \
        + kappa_y * (matrices[(x_2, y_1, y_1)] + matrices[(x_2, y_2, y_2)])

    term_c = kappa_x.T * (matrices[(y_1, x_1, x_1)] + matrices[(y_1, x_2, x_2)]) \
        + kappa_y.T * (matrices[(y_2, x_1, x_1)] + matrices[(y_2, x_2, x_2)])

    term_d = matrices[(x_1, x_1, y_1, y_1)] + matrices[(x_1,x_1,y_2,y_2)]\
        + matrices[(x_2, x_2, y_1, y_1)] + matrices[(x_2,x_2,y_2,y_2)]

    return np.exp(kappa).dot(np.exp(kappa.T)) * (term_a + term_b + term_c + term_d)

def canonical_A(interior, matrices, symbols, kappa, kappa_x, kappa_y):
        x_1, x_2 = symbols[0]
        shape = (interior.shape[0], 1)

        kappa = kappa[:interior.shape[0]].reshape(shape)
        kappa_x = kappa_x[:interior.shape[0]].reshape(shape)
        kappa_y = kappa_y[:interior.shape[0]].reshape(shape)

        term_a = kappa_x * matrices[(x_1)] + kappa_y * matrices[(x_2)]
        term_b = matrices[(x_1, x_1)] + matrices[(x_2, x_2)]

        return np.exp(kappa) * (term_a + term_b)