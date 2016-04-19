import sympy as sp
import bayesian_pdes as bpdes
from bayesian_pdes.problems import eit
from bayesian_pdes.problems.util import NamedLambda
import numpy as np
import unittest
import itertools


def construct_shell(radii, r_spacing=None):
    if r_spacing is None:
        r_spacing = radii[1] - radii[0]
    coords = [np.array([[0., 0.]])]
    for r in radii:
        # at each 'shell' we want a roughly equal number of theta around the diameter.
        # each theta should be about r_spacing apart
        n_theta = np.round(2*np.pi*r / r_spacing)
        thetas = np.linspace(0, 2*np.pi, n_theta+1)[:-1]
        x = np.cos(thetas)
        y = np.sin(thetas)
        coords.append(r*np.c_[x,y])
    coords = np.concatenate(coords)
    return coords


def test_function(length_scale, sigma, sigma_bar):

    x_1, x_2, y_1, y_2 = sp.symbols('x_1 x_2 y_1 y_2')
    k_sqexp = sp.exp(-((x_1 - y_1)**2 + (x_2 - y_2)**2) / (2*length_scale**2))
    exp_sigma = sp.exp(sigma)
    exp_sigma_bar = sp.exp(sigma_bar)

    A = NamedLambda(lambda k: (exp_sigma*k.diff(x_1)).diff(x_1) + (exp_sigma*k.diff(x_2)).diff(x_2), 'A')
    A_bar = NamedLambda(lambda k: (exp_sigma_bar*k.diff(y_1)).diff(y_1) + (exp_sigma_bar*k.diff(y_2)).diff(y_2), 'A_bar')

    B = NamedLambda(lambda k: exp_sigma*(k.diff(x_1)*x_1 + k.diff(x_2)*x_2), 'B')
    B_bar = NamedLambda(lambda k: exp_sigma_bar*(k.diff(y_1)*y_1 + k.diff(y_2)*y_2), 'B_bar')

    ops = [A, B]
    ops_bar = [A_bar, B_bar]
    oc = bpdes.operator_compilation.compile_sympy(
        ops,
        ops_bar,
        k_sqexp,
        [[x_1, x_2], [y_1, y_2]],
        mode='cython')

    return ops, ops_bar, oc


def compare_mats(sigma_fact, interior, bdy, length_scale):
    x_1, x_2, y_1, y_2 = sp.symbols('x_1 x_2 y_1 y_2')
    sigma = sigma_fact(x_1, x_2)
    sigma_bar = sigma_fact(y_1, y_2)

    k_sqexp = sp.exp(-((x_1 - y_1)**2 + (x_2 - y_2)**2) / (2*length_scale**2))

    kappa_int = np.vectorize(sp.lambdify([x_1, x_2], sigma))(interior[:, 0], interior[:, 1])
    kappa_bdy = np.vectorize(sp.lambdify([x_1, x_2], sigma))(bdy[:, 0], bdy[:, 1])
    kappa_x = np.vectorize(sp.lambdify([x_1, x_2], sigma.diff(x_1)))(interior[:, 0], interior[:, 1])
    kappa_y = np.vectorize(sp.lambdify([x_1, x_2], sigma.diff(x_2)))(interior[:, 0], interior[:, 1])

    fact = eit.EITFactory(k_sqexp, [x_1, x_2], [y_1, y_2], None, 1)
    my_oc = fact.get_operator_system(kappa_int, kappa_bdy, kappa_x, kappa_y)

    ops_test, ops_bar_test, oc_test = test_function(length_scale, sigma, sigma_bar)
    test_pts = np.r_[interior, bdy]

    for o1, o2, pts in zip(
            my_oc.operators,
            ops_test,
            [interior, bdy]):
        my_mat = my_oc[o1](pts, test_pts)
        base = oc_test[o2](pts, test_pts)
        np.testing.assert_almost_equal(my_mat, base, err_msg='Failed for {} vs {}'.format(o1, o2))

    for p1, p2, pts in zip(
            itertools.product(my_oc.operators, my_oc.operators_bar),
            itertools.product(ops_test, ops_bar_test),
            itertools.product([interior, bdy], [interior, bdy])):
        my_mat = my_oc[p1](pts[0], pts[1])
        base = oc_test[p2](pts[0], pts[1])
        np.testing.assert_almost_equal(my_mat, base, err_msg='Failed for {} vs {}'.format(p1, p2))


class EitSystemTest(unittest.TestCase):
    def test_matching_field(self):
        rs = np.linspace(0,1,6)[1:-1]
        interior = construct_shell(rs)
        bdy = construct_shell(np.array([1.]), 2*np.pi / 32.)
        #print(interior)

        compare_mats(lambda x, y: sp.Number(0), interior, bdy, 0.2)
        compare_mats(lambda x, y: (x+y)**2, interior, bdy, 0.2)
        compare_mats(lambda x, y: sp.cos(2*sp.pi*x)*sp.cos(2*sp.pi*y), interior, bdy, 0.2)