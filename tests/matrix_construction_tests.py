import unittest
from autograd import numpy as np
import numpy as true_np
import autograd
import sympy as sp
import bayesian_pdes as bpdes
from bayesian_pdes import pairwise

def k_autograd(x, x_prime, sigma):
    return np.exp(-(x-x_prime)**2 / (2*sigma**2))

k_x = autograd.grad(k_autograd, 0)
k_x_prime = autograd.grad(k_autograd, 1)
k_x_x_prime = autograd.grad(k_x, 1)


class MatrixConstructionTests(unittest.TestCase):
    def test_LLbar(self):
        x, y = sp.symbols('x y')
        Identity = lambda k: k
        A = lambda k: k.diff(x)
        A_bar = lambda k: k.diff(y)
        sigma = 0.1
        k = sp.exp(-(x-y)**2 / (2*sigma**2))

        ops = [A, Identity]
        ops_bar = [A_bar, Identity]

        oc = bpdes.operator_compilation.compile_sympy(ops, ops_bar, k, [[x], [y]], mode='cython')

        pts = np.linspace(0, 1, 11)

        interior = pts[1:-1, None]
        bdy = pts[[0,-1], None]

        print(interior.shape, bdy.shape)

        obs = [(interior, None), (bdy, None)]

        llbar = bpdes.collocation.calc_LLbar(ops, ops_bar, obs, oc)

        top_left = llbar[:len(interior), :len(interior)]
        compare_top_left = pairwise.apply(k_x_x_prime, interior, interior, np.array([sigma]))
        true_np.testing.assert_almost_equal(top_left, compare_top_left)

        bottom_right = llbar[len(interior):, len(interior):]
        compare_bottom_right = pairwise.apply(k_autograd, bdy, bdy, np.array([sigma]))
        true_np.testing.assert_almost_equal(bottom_right, compare_bottom_right)

        top_right = llbar[:len(interior), len(interior):]
        compare_top_right = pairwise.apply(k_x, interior, bdy, np.array([sigma]))
        true_np.testing.assert_almost_equal(top_right, compare_top_right)

        bottom_left = llbar[len(interior):, :len(interior)]
        compare_bottom_left = pairwise.apply(k_x_prime, bdy, interior, np.array([sigma]))
        true_np.testing.assert_almost_equal(bottom_left, compare_bottom_left)
