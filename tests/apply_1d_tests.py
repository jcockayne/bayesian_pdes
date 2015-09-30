__author__ = 'benorn'
from bayesian_pdes.collocation import apply_1d, calc_LLbar
import unittest
import sympy as sp
import numpy as np

class SympyHelpersTest(unittest.TestCase):
    def test_1d(self):
        x, y = sp.symbols('x y')
        f = x - y
        o = lambda expr: expr  # identity

        obs = [
            np.array([[1.], [2.]])
        ]
        grid_fun = apply_1d([o], f, [[x], [y]], obs)

        test_points = np.array([[2.], [1.], [0.]])

        res = grid_fun(test_points)
        expected = np.array([[1., 0., -1.], [0., -1., -2.]])

        self.assertEquals(res.shape, (2, 3))
        np.testing.assert_array_equal(res, expected)

    def test_2d(self):
        x_1, x_2, y_1, y_2 = sp.symbols('x_1 x_2 y_1 y_2')
        f = (x_1 - y_1) + (x_2 - y_2)

        o = lambda expr: expr  # identity

        obs = [
            np.array([[1., 1.], [2., 2.]])
        ]
        grid_fun = apply_1d([o], f, [[x_1, x_2], [y_1, y_2]], obs)

        test_points = np.array([[2., 2.], [1., 1.], [0., 0.]])

        res = grid_fun(test_points)
        expected = np.array([[2., 0., -2.], [0., -2., -4.]])

        self.assertEquals(res.shape, (2, 3))
        np.testing.assert_array_equal(res, expected)

    def test_1d_multi_op(self):
        x_1, x_2, y_1, y_2 = sp.symbols('x_1 x_2 y_1 y_2')
        f = (x_1 - y_1) + (x_2 - y_2)

        o1 = lambda expr: expr  # identity
        o2 = lambda expr: -expr

        obs = np.array([[1., 1.], [2., 2.]])
        grid_fun = apply_1d([o1, o2], f, [[x_1, x_2], [y_1, y_2]], [obs, obs])

        test_points = np.array([[2., 2.], [1., 1.], [0., 0.]])

        res = grid_fun(test_points)
        expected = np.array([
            [2., 0., -2.],
            [0., -2., -4.],
            [-2., 0., 2.],
            [0., 2., 4.]
        ])

        self.assertEquals(res.shape, (4, 3))
        np.testing.assert_array_equal(res, expected)

    def test_calc_LLbar(self):
        x, y = sp.symbols('x y')
        f = x - y

        o1 = lambda expr: expr  # identity
        o2 = lambda expr: -expr

        obs = [
            (np.array([[1.], [2.]]), None),
            (np.array([[-1.], [-2.]]), None)
            ]

        res = calc_LLbar([o1, o2], [o1, o2], f, [[x], [y]], obs)

        expected = np.array([
            [0., -1., -2., -3.],
            [1., 0., -3., -4.],
            [2., 3., 0., 1.],
            [3., 4., -1., 0.]
        ])

        self.assertEquals(res.shape, (4, 4))
        np.testing.assert_array_equal(res, expected)