import unittest
from bayesian_pdes.problems import laplacian_natural_kernel
import numpy as np
import scipy.integrate


def Lambda(z_1, z_2, epsilon):
    ret = np.maximum(1 - epsilon*np.abs(z_1-z_2), 0)
    return ret**2

# todo: check
def G(x_1, x_2):
    if x_1 >= x_2:
        return x_2*(x_1 - 1)
    return x_1*(x_2 - 1)


def I_1_quadrature(x_1, x_2, epsilon):
    def integrand(z_2, z_1):
        return z_1*z_2*Lambda(z_1, z_2, epsilon)
    return scipy.integrate.dblquad(integrand, 0, x_1, lambda _: 0, lambda _: x_2)


def I_2_quadrature(x_1, x_2, epsilon):
    def integrand(z_2, z_1):
        return z_1*(z_2-1)*Lambda(z_1, z_2, epsilon)
    return scipy.integrate.dblquad(integrand, 0, x_1, lambda _: x_2, lambda _: 1)


def I_3_quadrature(x_1, x_2, epsilon):
    eps_inv = 1./epsilon
    def integrand(z_2, z_1):
        return (z_1-1)*z_2*Lambda(z_1, z_2, epsilon)
    return scipy.integrate.dblquad(integrand, x_1, 1, lambda z_2: max(0, z_2-eps_inv), lambda z_2: x_2)
    #return scipy.integrate.dblquad(integrand, x_1, 1, lambda z_2: 0, lambda z_2: x_2)

def I_4_quadrature(x_1, x_2, epsilon):
    def integrand(z_2, z_1):
        return (z_1-1)*(z_2-1)*Lambda(z_1, z_2, epsilon)
    return scipy.integrate.dblquad(integrand, x_1, 1, lambda z_2: x_2, lambda z_2: 1)
    #return scipy.integrate.dblquad(integrand, x_1, 1, lambda z_2: 0, lambda z_2: x_2)

def kern_quadrature(x_1, x_2, epsilon):
    def integrand(z_2, z_1):
        return G(x_1, z_1)*G(x_2, z_2)*Lambda(z_1, z_2, epsilon)
    return scipy.integrate.dblquad(integrand, 0, 1, lambda z_2: 0, lambda z_2: 1)

def A_k_quadrature(x_1, x_2, epsilon):
    def integrand(z_2):
        return G(x_2, z_2) * Lambda(x_1, z_2, epsilon)
    return scipy.integrate.quad(integrand, 0, 1)

def A_bar_k_quadrature(x_1, x_2, epsilon):
    def integrand(z_1):
        return G(x_1, z_1) * Lambda(x_2, z_1, epsilon)
    return scipy.integrate.quad(integrand, 0, 1)


class LaplacianNaturalKernelTests(unittest.TestCase):
    # close but are we close enough??
    def test_I_1(self):
        # exclude points directly on the boundary
        pts = np.linspace(0, 1, 11)[1:-1]
        for eps in [5., 10.]:
            natural_kernel = laplacian_natural_kernel.LaplacianNaturalKernel(eps, True)
            for x in pts:
                for y in pts[pts >= x]:
                    res = natural_kernel.I_1(x, y)
                    res_quad, err = I_1_quadrature(x, y, eps)
                    np.testing.assert_almost_equal(res, res_quad, err_msg='Failed at ({}, {}), epsilon={}'.format(x, y, eps))

    def test_I_2(self):
        # exclude points directly on the boundary
        pts = np.linspace(0, 1, 21)[1:-1]
        for eps in [5., 10.]:
            natural_kernel = laplacian_natural_kernel.LaplacianNaturalKernel(eps, True)
            for x in pts:
                for y in pts[pts >= x]:
                    res = natural_kernel.I_2(x, y)
                    res_quad, err = I_2_quadrature(x, y, eps)
                    np.testing.assert_almost_equal(res, res_quad, err_msg='Failed at ({}, {}), epsilon={}'.format(x, y, eps))

    def test_I_3(self):
        # exclude points directly on the boundary
        for eps in [5., 10.]:
            pts = np.linspace(0, 1, 21)[1:-1]
            natural_kernel = laplacian_natural_kernel.LaplacianNaturalKernel(eps, True)
            for x in pts:
                for y in pts[(pts >= x)]:
                    res = natural_kernel.I_3(x, y)
                    res_quad, err = I_3_quadrature(x, y, eps)
                    np.testing.assert_almost_equal(res, res_quad, err_msg='Failed at ({}, {}), epsilon={}'.format(x, y, eps))

    def test_I_4(self):
        # exclude points directly on the boundary
        for eps in [5., 10.]:
            pts = np.linspace(0, 1, 21)[1:-1]
            natural_kernel = laplacian_natural_kernel.LaplacianNaturalKernel(eps, True)
            for x in pts:
                for y in pts[(pts >= x)]:
                    res = natural_kernel.I_4(x, y)
                    res_quad, err = I_4_quadrature(x, y, eps)
                    np.testing.assert_almost_equal(res, res_quad, err_msg='Failed at ({}, {}), epsilon={}'.format(x, y, eps))

    def test_kern(self):
        for eps in [5., 10.]:
            pts = np.linspace(0, 1, 21)[1:-1]
            natural_kernel = laplacian_natural_kernel.LaplacianNaturalKernel(eps, True)
            for x in pts:
                for y in pts:
                    res = natural_kernel.kern(x, y)
                    res_quad, err = kern_quadrature(x, y, eps)
                    np.testing.assert_almost_equal(res, res_quad, err_msg='Failed at ({}, {}), epsilon={}'.format(x, y, eps))

    def test_A_k(self):
        for eps in [5., 10.]:
            pts = np.linspace(0, 1, 21)[1:-1]
            natural_kernel = laplacian_natural_kernel.LaplacianNaturalKernel(eps, True)
            for x in pts:
                for y in pts:
                    res = natural_kernel.A_k(x, y)
                    res_quad, err = A_k_quadrature(x, y, eps)
                    np.testing.assert_almost_equal(res, res_quad, err_msg='Failed at ({}, {}), epsilon={}'.format(x, y, eps))

    def test_A_bar_k(self):
        for eps in [5., 10.]:
            pts = np.linspace(0, 1, 21)[1:-1]
            natural_kernel = laplacian_natural_kernel.LaplacianNaturalKernel(eps, True)
            for x in pts:
                for y in pts:
                    res = natural_kernel.A_bar_k(x, y)
                    res_quad, err = A_bar_k_quadrature(x, y, eps)
                    np.testing.assert_almost_equal(res, res_quad, err_msg='Failed at ({}, {}), epsilon={}'.format(x, y, eps))

