import sympy as sp
import numpy as np


class LaplacianNaturalKernel(object):
    def __init__(self, epsilon, debug=False):
        self.epsilon = epsilon
        modules = [
            {'Min': min, 'Max': max},
            'numpy'
        ]

        x_1, x_2, z_1, z_2 = sp.symbols('x_1 x_2 z_1 z_2')
        eps_inv = 1./epsilon

        I_1_left_integrand = z_1 * z_2 * (1 - epsilon * (z_2-z_1))**2
        I_1_right_integrand = z_1 * z_2 * (1 - epsilon * (z_1-z_2))**2

        
        I_1_1_left = sp.integrate(I_1_left_integrand, (z_1, 0, z_2))
        I_1_1_right = sp.integrate(I_1_right_integrand, (z_1, z_2, eps_inv))
        I_1_1 = sp.integrate(I_1_1_left + I_1_1_right, (z_2, 0, sp.Min(z_1+eps_inv, x_2)))

        # case 1: x_1 < x_2 - epsilon^{-1}
        I_1_1_c1_right = sp.integrate(I_1_right_integrand, (z_2, 0, z_1))
        I_1_1_c1_left = sp.integrate(I_1_left_integrand, (z_2, z_1, z_1+eps_inv))
        I_1_1_c1 = sp.integrate(I_1_1_c1_left + I_1_1_c1_right, (z_1, 0, eps_inv))

        I_1_2_c1_right = sp.integrate(I_1_right_integrand, (z_2, z_1-eps_inv, z_1))
        I_1_2_c1_left = sp.integrate(I_1_left_integrand, (z_2, z_1, z_1+eps_inv))
        I_1_2_c1 = sp.integrate(I_1_2_c1_left + I_1_2_c1_right, (z_1, eps_inv, x_1))
        self.I_1_c1 = sp.lambdify([x_1, x_2], I_1_1_c1 + I_1_2_c1, modules=modules)

        # case 2: x_1 > x_2 - epsilon^{-1}
        I_1_1_c2 = I_1_1_c1
        I_1_2_c2_right = sp.integrate(I_1_right_integrand, (z_2, z_1-eps_inv, z_1))
        I_1_2_c2_left = sp.integrate(I_1_left_integrand, (z_2, z_1, z_1+eps_inv))
        I_1_2_c2 = sp.integrate(I_1_2_c2_left + I_1_2_c2_right, (z_1, eps_inv, x_2-eps_inv))

        I_1_3_c2_right = sp.integrate(I_1_right_integrand, (z_2, z_1-eps_inv, z_1))
        I_1_3_c2_left = sp.integrate(I_1_left_integrand, (z_2, z_1, x_2))
        I_1_3_c2 = sp.integrate(I_1_3_c2_right + I_1_3_c2_left, (z_1, x_2-eps_inv, x_1))

        self.I_1_c2 = sp.lambdify([x_1, x_2], I_1_1_c2 + I_1_2_c2 + I_1_3_c2, modules=modules)

        I_2_left_integrand = z_1 * (z_2-1) * (1 - epsilon * (z_2-z_1))**2

        I_2_left_c1 = sp.integrate(I_2_left_integrand, (z_2, x_2, z_1+eps_inv))
        I_2_c1 = sp.integrate(I_2_left_c1, (z_1, sp.Max(x_2-eps_inv, 0.), x_1))

        I_2_left_c2 = sp.integrate(I_2_left_integrand, (z_1, z_2-eps_inv, x_1))
        I_2_c2 = sp.integrate(I_2_left_c2, (z_2, x_2, sp.Min(x_1+eps_inv, 1.)))

        self.I_2_c1 = sp.lambdify([x_1, x_2], I_2_c1, modules=modules)
        self.I_2_c2 = sp.lambdify([x_1, x_2], I_2_c2, modules=modules)

        I_3_right_integrand = (z_1-1) * z_2 * (1 - epsilon * (z_1-z_2))**2
        I_3_left_integrand = (z_1-1) * z_2 * (1 - epsilon * (z_2-z_1))**2

        # assuming x_2 + eps_inv < 1...
        I_3_1_c1_right = sp.integrate(I_3_right_integrand, (z_2, z_1-eps_inv, z_1))
        I_3_1_c1_left = sp.integrate(I_3_left_integrand, (z_2, z_1, z_1+eps_inv))
        I_3_1_c1 = sp.integrate(I_3_1_c1_right + I_3_1_c1_left, (z_1, x_1, x_2-eps_inv))
        # sp.Max(x_1, eps_inv)
        I_3_2_c1_right = sp.integrate(I_3_right_integrand, (z_2, z_1-eps_inv, z_1))
        I_3_2_c1_left = sp.integrate(I_3_left_integrand, (z_2, z_1, x_2))
        I_3_2_c1 = sp.integrate(I_3_2_c1_right + I_3_2_c1_left, (z_1, sp.Max(x_2-eps_inv, x_1), x_2))

        I_3_3_c1_right = sp.integrate(I_3_right_integrand, (z_2, z_1-eps_inv, x_2))
        I_3_3_c1 = sp.integrate(I_3_3_c1_right, (z_1, x_2, sp.Min(x_2+eps_inv, 1.)))

        I_3_4_c1_right = sp.integrate(I_3_right_integrand, (z_2, z_1-eps_inv, 0))
        I_3_4_c1 = sp.integrate(I_3_4_c1_right, (z_1, x_1, eps_inv))

        self.I_3_1_c1 = sp.lambdify([x_1, x_2], I_3_1_c1, modules=modules)
        self.I_3_2_c1 = sp.lambdify([x_1, x_2], I_3_2_c1, modules=modules)
        self.I_3_3_c1 = sp.lambdify([x_1, x_2], I_3_3_c1, modules=modules)
        self.I_3_4 = sp.lambdify([x_1, x_2], I_3_4_c1, modules=modules)

    def I_1(self, x_1, x_2):
        eps_inv = 1./self.epsilon
        assert eps_inv < 1, 'We assume non-global support!'
        if x_1 > x_2:
            raise Exception('We are assuming x_1 < x_2')
        if x_1 < x_2 - eps_inv:
            return self.I_1_c1(x_1, x_2)
        elif x_1 <= x_2:
            return self.I_1_c2(x_1, x_2)
        else:
            raise Exception('Case not understood!')

    def I_2(self, x_1, x_2):
        eps_inv = 1./self.epsilon
        if x_1 > x_2:
            raise Exception('We are assuming x_1 < x_2')
        if x_2 > x_1 + eps_inv:
            return 0.0
        elif x_1 + eps_inv > 1.:
            return self.I_2_c2(x_1, x_2)
        return self.I_2_c1(x_1, x_2)

    def I_3(self, x_1, x_2):
        eps_inv = 1./self.epsilon
        if x_1 > x_2:
            raise Exception('We are assuming x_1 < x_2')
        res = self.I_3_2_c1(x_1, x_2) + self.I_3_3_c1(x_1, x_2)
        if x_2 - eps_inv > x_1:
            res += self.I_3_1_c1(x_1, x_2)
        if x_1 < eps_inv:
            res -= self.I_3_4(x_1, x_2)
        return res
