__author__ = 'benorn'
import numpy as np
import bayesian_pdes
import sympy as sp
import unittest


class CollocateTests(unittest.TestCase):
    def test_collocate_2d(self):
        x_1,x_2,y_1,y_2 = sp.symbols('x_1 x_2 y_1 y_2')
        length_scale = 0.25
        k = sp.exp(-((x_1-y_1)**2+(x_2-y_2)**2) / (2*length_scale**2))
        A = lambda f: sp.diff(f, x_1, x_1) + sp.diff(f, x_2, x_2)
        Abar = lambda f: sp.diff(f, y_1, y_1) + sp.diff(f, y_2, y_2)
        B = lambda f: f
        Bbar = lambda f: f

        # interior observations: sin(x)*sin(y)
        x_interior, y_interior = np.mgrid[0.1:0.9:9j, 0.1:0.9:9j]
        interior = np.c_[x_interior.ravel(), y_interior.ravel()]
        exterior = np.r_[
            np.c_[np.linspace(0.1,0.9,9), np.zeros(9)],
            np.c_[np.linspace(0.1,0.9,9), np.ones(9)],
            np.c_[np.zeros(9), np.linspace(0.1,0.9,9)],
            np.c_[np.ones(9), np.linspace(0.1,0.9,9)]
        ]

        interior_obs = np.sin(2*np.pi*interior[:,0]) + np.sin(2*np.pi*interior[:,1])
        exterior_obs = np.r_[
            np.zeros(18),
            np.zeros(18)
        ]

        print interior.shape, interior_obs.shape
        print exterior.shape, exterior_obs.shape
        posterior = bayesian_pdes.collocate(
            [A, B],
            [Abar, Bbar],
            k,
            [[x_1, x_2], [y_1, y_2]],
            [(interior, interior_obs), (exterior, exterior_obs)]
        )

        # todo: assert something instead of just marvelling at the fact that this works.
        mean, cov = posterior(interior)
