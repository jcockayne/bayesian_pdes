__author__ = 'benorn'
import numpy as np
import bayesian_pdes
import sympy as sp
import unittest


class CollocateTests(unittest.TestCase):

    def test_collocate_1d(self):
        x, y = sp.symbols('x y')
        length_scale = 0.05
        k = sp.exp(-((x-y)**2) / (2*length_scale**2))
        A = lambda f: sp.diff(f, x, x)
        Abar = lambda f: sp.diff(f, y, y)
        B = lambda f: f
        Bbar = lambda f: f

        # interior observations: sin(x)*sin(y)
        interior = np.linspace(0,1)[1:-1, None]
        exterior = np.array([0.,1.])[:, None]

        interior_obs = np.sin(2*np.pi*interior)
        exterior_obs = np.array([0.,0.])[:, None]

        print interior.shape, interior_obs.shape
        print exterior.shape, exterior_obs.shape

        op_system = bayesian_pdes.operator_compilation.compile_sympy([A, B], [Abar, Bbar], k, [[x], [y]])
        posterior = bayesian_pdes.collocate(
            [A, B],
            [Abar, Bbar],
            [(interior, interior_obs), (exterior, exterior_obs)],
            op_system
        )

        mean, cov = posterior(interior)
        actual = -1 * np.sin(2*np.pi*interior) / (4*np.pi**2)

        ci = cov.dot(0.25*np.ones_like(mean))  # represents a 0.25SD CI for each observation

        err = np.abs(mean - actual)
        np.testing.assert_array_less(err, ci)

    def test_collocate_2d(self):
        x_1,x_2,y_1,y_2 = sp.symbols('x_1 x_2 y_1 y_2')
        length_scale = 0.05
        k = sp.exp(-((x_1-y_1)**2+(x_2-y_2)**2) / (2*length_scale**2))
        A = lambda f: sp.diff(f, x_1, x_1) + sp.diff(f, x_2, x_2)
        Abar = lambda f: sp.diff(f, y_1, y_1) + sp.diff(f, y_2, y_2)
        B = lambda f: f
        Bbar = lambda f: f

        x_interior, y_interior = np.mgrid[0:1:20j, 0:1:20j]
        x_interior = x_interior.ravel(); y_interior = y_interior.ravel()

        on_bdy = (x_interior == 0) | (x_interior == 1) | (y_interior == 0) | (y_interior == 1)

        interior = np.c_[x_interior[~on_bdy], y_interior[~on_bdy]]
        exterior = np.c_[x_interior[on_bdy], y_interior[on_bdy]]

        interior_obs = np.sin(2*np.pi*interior[:,0]) + np.sin(2*np.pi*interior[:,1])
        exterior_obs = np.zeros(exterior.shape[0])

        print interior.shape, interior_obs.shape
        print exterior.shape, exterior_obs.shape

        op_system = bayesian_pdes.operator_compilation.compile_sympy([A, B], [Abar, Bbar], k, [[x_1, x_2], [y_1, y_2]])
        posterior = bayesian_pdes.collocate(
            [A, B],
            [Abar, Bbar],
            [(interior, interior_obs), (exterior, exterior_obs)],
            op_system
        )

        import time
        start = time.time()

        # test_x, test_y = np.mgrid[0:1:21j, 0:1:21j]
        # test_points = np.c_[test_x.ravel(), test_y.ravel()]
        test_points = interior
        mean, cov = posterior(test_points)

        end = time.time()
        print end - start

        actual = - (np.sin(2*np.pi*test_points[:,0]) + np.sin(2*np.pi*test_points[:,1])) / (4*np.pi**2)
        ci = cov.dot(0.25*np.ones_like(mean))  # represents a 0.25SD CI for each observation

        err = np.abs(mean - actual)
        np.testing.assert_array_less(err, ci)

