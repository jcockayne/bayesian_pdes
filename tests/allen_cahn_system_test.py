import unittest
import sympy as sp
import bayesian_pdes as bpdes
from bayesian_pdes.problems import allen_cahn
import numpy as np
import itertools

class AllenCahnSystemTest(unittest.TestCase):
    def test_matrices(self):
        length_scale = 0.2
        delta = 0.04

        x_1, x_2, y_1, y_2 = sp.symbols('x_1 x_2 y_1 y_2')
        k_sqexp = sp.exp(-((x_1 - y_1)**2 + (x_2 - y_2)**2) / (2*length_scale**2))

        A_1 = lambda k: -delta*(k.diff(x_1, x_1) + k.diff(x_2, x_2)) - 1./delta * k
        A_1_bar = lambda k: -delta*(k.diff(y_1, y_1) + k.diff(y_2, y_2)) - 1./delta * k
        A_2 = lambda k: k
        A_2_bar = lambda k: k
        B_1 = lambda k: k
        B_1_bar = lambda k: k

        ops = [A_1, A_2, B_1]
        ops_bar = [A_1_bar, A_2_bar, B_1_bar]

        compare_op_system = bpdes.operator_compilation.compile_sympy(
            ops,
            ops_bar,
            k_sqexp,
            [[x_1, x_2], [y_1, y_2]],
            mode='cython'
        )

        op_system_factory = allen_cahn.AllenCahnFactory(k_sqexp, [x_1, x_2], [y_1, y_2], None)
        my_op_system = op_system_factory.get_operator_system(delta, 1)

        true_ops = my_op_system.operators
        true_ops_bar = my_op_system.operators_bar

        test_x, test_y = np.mgrid[0:1:11j, 0:1:11j]
        test_x = test_x.ravel()
        test_y = test_y.ravel()
        test = np.column_stack([test_x, test_y])
        for op, compare_op in zip(true_ops, ops):
            my_fun = my_op_system[op]
            compare_fun = compare_op_system[compare_op]

            compare_mat = compare_fun(test, test)
            my_mat = my_fun(test, test, np.array([length_scale]))

            np.testing.assert_almost_equal(my_mat, compare_mat, err_msg='Failed for {} vs. {}'.format(op, compare_op))

        for op, compare_op in zip(true_ops_bar, ops_bar):
            my_fun = my_op_system[op]
            compare_fun = compare_op_system[compare_op]

            compare_mat = compare_fun(test, test)
            my_mat = my_fun(test, test, np.array([length_scale]))

            np.testing.assert_almost_equal(my_mat, compare_mat, err_msg='Failed for {} vs. {}'.format(op, compare_op))

        for comb_1, comb_2 in zip(itertools.product(true_ops, true_ops_bar), itertools.product(ops, ops_bar)):
            my_fun = my_op_system[comb_1]
            compare_fun = compare_op_system[comb_2]

            compare_mat = compare_fun(test, test)
            my_mat = my_fun(test, test, np.array([length_scale]))

            np.testing.assert_almost_equal(my_mat, compare_mat, err_msg='Failed for {} vs. {}'.format(comb_1, comb_2))




