import unittest
import numpy as np
import bayesian_pdes.inversion as inv


def random_system(size, n_rhs=1):
    mat = np.random.normal(size=(size,size))
    matsq = mat.dot(mat.T)

    soln = np.random.normal(size=(size,n_rhs))
    rhs = matsq.dot(soln)

    return matsq, rhs, soln


class InversionTests(unittest.TestCase):
    def test_direct_inversion(self):
        mat, rhs, soln = random_system(10)

        inverter = inv.DirectInversion(mat)
        inverter_soln = inverter.apply(rhs)

        np.testing.assert_almost_equal(inverter_soln, soln)

    def test_cg_inversion(self):
        mat, rhs, soln = random_system(10)
        tol = 1e-5

        inverter = inv.CGInversion(mat, tol)
        inverter_soln = inverter.apply(rhs)

        np.testing.assert_allclose(inverter_soln, soln, atol=tol)

    def test_cg_inversion_left(self):
        size = 10
        n_rhs = 2
        mat = np.random.normal(size=(size,size))
        mat = mat.dot(mat.T)

        soln = np.random.normal(size=(n_rhs,size))
        lhs = soln.dot(mat)
        tol = 1e-5

        inverter = inv.CGInversion(mat, tol)
        inverter_soln = inverter.apply_left(lhs)

        np.testing.assert_allclose(inverter_soln, soln, atol=tol)


    def test_direct_inversion_left(self):
        size = 10
        n_rhs = 2
        mat = np.random.normal(size=(size,size))
        mat = mat.dot(mat.T)

        soln = np.random.normal(size=(n_rhs,size))
        lhs = soln.dot(mat)
        tol = 1e-5

        inverter = inv.DirectInversion(mat)
        inverter_soln = inverter.apply_left(lhs)

        np.testing.assert_allclose(inverter_soln, soln, atol=tol)


    def test_direct_inversion_multi(self):
        mat, rhs, soln = random_system(10,10)

        inverter = inv.DirectInversion(mat)
        inverter_soln = inverter.apply(rhs)

        np.testing.assert_almost_equal(inverter_soln, soln)

    def test_cg_inversion_mult(self):
        mat, rhs, soln = random_system(10,10)
        tol = 1e-5

        inverter = inv.CGInversion(mat, tol)
        inverter_soln = inverter.apply(rhs)

        np.testing.assert_allclose(inverter_soln, soln, atol=tol)