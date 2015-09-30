__author__ = 'benorn'
from bayesian_pdes.sympy_helpers import sympy_function
import unittest
import sympy as sp
import numpy as np

class SympyHelpersTest(unittest.TestCase):
    def test_1d(self):
        x = sp.Symbol('x')
        f = x**2

        f_fun = sympy_function(f, [x])

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