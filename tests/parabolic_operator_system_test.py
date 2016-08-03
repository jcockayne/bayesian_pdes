import unittest
import sympy as sp
from bayesian_pdes.operator_compilation.sympy_gram import compile_sympy
import numpy as np
import itertools
from bayesian_pdes.parabolic.discretised import ParabolicOperatorSystem

class ParabolicOperatorSystemTest(unittest.TestCase):
    def test_matrices(self):
        x, y = sp.symbols('x,y')
        dt = sp.symbols('dt')
        theta = sp.symbols('theta')

        def A(k):
            return k.diff(x,x)
        def A_bar(k):
            return k.diff(y,y)

        def L_explicit(k):
            return k + theta*dt*A(k)
        def L_bar_explicit(k):
            return k + theta*dt*A(k)
        def L_implicit(k):
            return k - (1-theta)*dt*A(k)
        def L_bar_implicit(k):
            return k - (1-theta)*dt*A(k)

        def neumann(k):
            return k.diff(x)
        def neumann_bar(k):
            return k.diff(y)

        k = sp.exp(-(x-y)**2)
        base_op_system = compile_sympy([L_explicit, L_implicit, neumann],
                                       [L_bar_explicit, L_bar_implicit, neumann_bar],
                                       k,
                                       [[x], [y], [dt, theta]]
                                       )
        base_compare = compile_sympy([A, neumann], [A_bar, neumann_bar], k, [[x], [y]])

        test_pts = np.linspace(0,1,51)[:, None]

        def compare_matrices(base_key, parabolic_key):
            base_mat = base_op_system[base_key](test_pts, test_pts, fun_args)
            test_mat = parabolic_op_system[parabolic_key](test_pts, test_pts, np.array([]))
            np.testing.assert_almost_equal(base_mat, test_mat, err_msg='Failed for {} vs. {}'.format(base_key, parabolic_key))

        for theta_val in [0., 0.5, 1.]:
            dt_val = 0.1
            parabolic_op_system = ParabolicOperatorSystem(base_compare, A, A_bar, theta_val, dt_val)

            association = {
                (): (),
                L_explicit: parabolic_op_system.explicit_op,
                L_bar_explicit: parabolic_op_system.explicit_op_bar,
                L_implicit: parabolic_op_system.implicit_op,
                L_bar_implicit: parabolic_op_system.implicit_op_bar,
                neumann: neumann,
                neumann_bar: neumann_bar
            }
            fun_args = np.array([dt_val, theta_val])
            for k in [(), L_explicit, L_bar_explicit, L_implicit, L_bar_implicit, neumann, neumann_bar]:
                left_key = (k,)
                right_key = (association[k], )
                compare_matrices(left_key, right_key)
            for k1, k2 in itertools.product([(), L_explicit, L_implicit, neumann], [(), L_bar_explicit, L_bar_implicit, neumann_bar]):
                left_key = (k1, k2)
                right_key = (association[k1], association[k2])
                compare_matrices(left_key, right_key)