import unittest
import sympy as sp
from bayesian_pdes import operator_compilation
import numpy as np


class SympyGramTests(unittest.TestCase):
    def test_sqexp_1d(self):
        x, y = sp.symbols('x,y')
        k = sp.exp(-(x-y)**2)
        A = lambda k: k.diff(x,x)
        Abar = lambda k: k.diff(y, y)

        base = operator_compilation.compile_sympy([A], [Abar], k, [[x], [y]], mode='cython', debug=True)
        compare = operator_compilation.sympy_gram.compile_sympy([A], [Abar], k, [[x], [y]])
        compare_caches(base, compare, 1)

    def test_sqexp_2d(self):
        x1, x2, y1, y2 = sp.symbols('x1,x2,y1,y2')
        k = sp.exp(-(x1-y1)**2 - (x2-y2)**2)
        A = lambda k: k.diff(x1,x1) + k.diff(x2, x2)
        Abar = lambda k: k.diff(y1, y1) + k.diff(y2, y2)

        base = operator_compilation.compile_sympy([A], [Abar], k, [[x1, x2], [y1, y2]], mode='cython', debug=True)
        compare = operator_compilation.sympy_gram.compile_sympy([A], [Abar], k, [[x1, x2], [y1, y2]])
        compare_caches(base, compare, 2)

    def test_sqexp_2d_extra_args(self):
        x1, x2, y1, y2 = sp.symbols('x1,x2,y1,y2')
        sigma = sp.symbols('sigma')
        k = sp.exp(-((x1-y1)**2 - (x2-y2)**2) / (2.*sigma**2))
        A = lambda k: k.diff(x1,x1) + k.diff(x2, x2)
        Abar = lambda k: k.diff(y1, y1) + k.diff(y2, y2)

        base = operator_compilation.compile_sympy([A], [Abar], k, [[x1, x2], [y1, y2], [sigma]], mode='cython', debug=True)
        compare = operator_compilation.sympy_gram.compile_sympy([A], [Abar], k, [[x1, x2], [y1, y2], [sigma]])
        compare_caches(base, compare, 2, np.array([0.1]))


def compare_caches(base, compare, ndim, extra=None):
    all = [()] + [(o,) for o in base.operators] + [(o,) for o in base.operators_bar] + \
          [(o1, o2) for o1 in base.operators for o2 in base.operators_bar]

    test_data = np.random.normal(size=(100,ndim))
    for item in all:
        base_mat = base[item](test_data, test_data, extra)
        compare_mat = compare[item](test_data, test_data, extra)

        np.testing.assert_allclose(base_mat, compare_mat)