import unittest
import sympy as sp
import bayesian_pdes
from bayesian_pdes.problems import laplacian_inverse_problem
import numpy as np

class LaplacianInverseProblemTest(unittest.TestCase):
    def test_one_op(self):
        x, y = sp.symbols('x y')
        theta = sp.symbols('theta')
        A = lambda k: k.diff(x, x)
        A_bar = lambda k: k.diff(y, y)

        k = sp.exp(-(x-y)**2 / 2.)

        base_system = bayesian_pdes.operator_compilation.compile_sympy([A], [A_bar], k, [[x], [y]])

        A_true = lambda k: theta*A(k)
        A_bar_true = lambda k: theta*A_bar(k)

        laplacian_system = laplacian_inverse_problem.LaplacianInverseProblemFactory(base_system, 1)

        true_system = bayesian_pdes.operator_compilation.compile_sympy([A_true], [A_bar_true], k, [[x], [y], [theta]])

        test_points = np.linspace(0, 1, 11)[:,None]

        for theta in np.linspace(-1, 1, 10):
            test_system = laplacian_system.get_operator_system(theta)
            test_mat = test_system[(test_system.A, )](test_points, test_points)
            true_mat = true_system[(A_true, )](test_points, test_points, np.array([theta]))
            np.testing.assert_almost_equal(test_mat, true_mat)

            test_mat = test_system[(test_system.A_bar, )](test_points, test_points)
            true_mat = true_system[(A_bar_true, )](test_points, test_points, np.array([theta]))
            np.testing.assert_almost_equal(test_mat, true_mat)

            test_mat = test_system[(test_system.A, test_system.A_bar)](test_points, test_points)
            true_mat = true_system[(A_true, A_bar_true)](test_points, test_points, np.array([theta]))
            np.testing.assert_almost_equal(test_mat, true_mat)