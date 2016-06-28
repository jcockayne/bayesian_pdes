__author__ = 'benorn'
from bayesian_pdes.operator_compilation.sympy_helpers import sympy_function
import unittest
import sympy as sp
from sympy.utilities.autowrap import autowrap
import numpy as np

class SympyHelpersTest(unittest.TestCase):
    def test_autowrap(self):
        x = sp.Symbol('x')
        f = x**2
        print 'yo'
        try:
            autowrap(f, backend='cython', tempdir='/Users/benorn/code_tmp', verbose=True)
        except Exception as ex:
            print ex
            import sys
            print sys.path

    def test_1d(self):
        x = sp.Symbol('x')
        f = x**2

        f_fun = sympy_function(f, [x], mode='cython', debug=True)

        self.assertEquals(f_fun(2.), 4.)
        self.assertEquals(f_fun(-2.), 4.)

        with self.assertRaises(AssertionError):
            f_fun([2., 2.], 2.)
        with self.assertRaises(AssertionError):
            f_fun(1., [2., 2.])

    def test_2d(self):
        x_1, x_2 = sp.symbols('x_1 x_2')
        f = x_1 * x_2

        f_fun = sympy_function(f, [[x_1, x_2]])

        self.assertEquals(f_fun([2., 2.]), 4.)
        self.assertEquals(f_fun(np.array([2., 2.])), 4.)
        self.assertEquals(f_fun(np.array([-2., 2.])), -4.)
        with self.assertRaises(AssertionError):
            f_fun([2., 2.], 2.)
        with self.assertRaises(AssertionError):
            f_fun(1., [2., 2.])

    def test_2d_2arg(self):
        x_1, x_2, y_1, y_2 = sp.symbols('x_1 x_2 y_1 y_2')
        f = (x_1 - y_1) + (x_2 - y_2)

        f_fun = sympy_function(f, [[x_1, x_2], [y_1, y_2]])

        self.assertEquals(f_fun([2., 2.], [1., 1.]), 2.)
        self.assertEquals(f_fun(np.array([2., 2.]), np.array([1.,1.])), 2.)
        with self.assertRaises(AssertionError):
            f_fun(2., 2.)
        with self.assertRaises(AssertionError):
            f_fun([2., 2.], 1.)
        with self.assertRaises(AssertionError):
            f_fun(1., [2., 2.])