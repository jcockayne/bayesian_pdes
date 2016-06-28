import sympy as sp
import numpy as np
from bayesian_pdes import pairwise


class LaplacianNaturalKernel(object):
    """
    If you are reading this you have found a monstrous piece of code, you poor soul.
    """
    def __init__(self, epsilon, debug=False):
        self.debug = debug
        self.epsilon = epsilon

        self.A = 'A'
        self.A_bar = 'A_bar'
        self.B = 'B'
        self.B_bar = 'B_bar'

        self.operators = [self.A]
        self.operators_bar = [self.A_bar]

        x_1, x_2, z_1, z_2 = sp.symbols('x_1 x_2 z_1 z_2')
        eps_inv = 1./epsilon

        I_1_left_integrand = z_1 * z_2 * (1 - epsilon * (z_2-z_1))**2
        I_1_right_integrand = z_1 * z_2 * (1 - epsilon * (z_1-z_2))**2

        I_1_1_right = sp.integrate(I_1_right_integrand, (z_2, 0, z_1))
        I_1_1_left = sp.integrate(I_1_left_integrand, (z_2, z_1, x_2))
        I_1_1 = sp.integrate(I_1_1_left + I_1_1_right, (z_1, 0, x_1))

        I_1_2_right = sp.integrate(I_1_right_integrand, (z_2, 0, z_1 - eps_inv))
        I_1_2 = sp.integrate(I_1_2_right, (z_1, eps_inv, x_1))

        I_1_3_left = sp.integrate(I_1_left_integrand, (z_1, 0, z_2 - eps_inv))
        I_1_3 = sp.integrate(I_1_3_left, (z_2, eps_inv, sp.Min(x_1 + eps_inv, x_2)))

        I_1_4_left = sp.integrate(I_1_left_integrand, (z_1, 0, x_1))
        I_1_4 = sp.integrate(I_1_4_left, (z_2, x_1+eps_inv, x_2))

        self.I_1_1 = self.functionise_sympy([x_1, x_2], I_1_1)
        self.I_1_2 = self.functionise_sympy([x_1, x_2], I_1_2)
        self.I_1_3 = self.functionise_sympy([x_1, x_2], I_1_3)
        self.I_1_4 = self.functionise_sympy([x_1, x_2], I_1_4)

        I_2_left_integrand = z_1 * (z_2-1) * (1 - epsilon * (z_2-z_1))**2

        I_2_left_c1 = sp.integrate(I_2_left_integrand, (z_2, x_2, z_1+eps_inv))
        I_2_c1 = sp.integrate(I_2_left_c1, (z_1, sp.Max(x_2-eps_inv, 0.), x_1))

        I_2_left_c2 = sp.integrate(I_2_left_integrand, (z_1, z_2-eps_inv, x_1))
        I_2_c2 = sp.integrate(I_2_left_c2, (z_2, x_2, sp.Min(x_1+eps_inv, 1.)))

        self.I_2_c1 = self.functionise_sympy([x_1, x_2], I_2_c1)
        self.I_2_c2 = self.functionise_sympy([x_1, x_2], I_2_c2)

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

        self.I_3_1_c1 = self.functionise_sympy([x_1, x_2], I_3_1_c1)
        self.I_3_2_c1 = self.functionise_sympy([x_1, x_2], I_3_2_c1)
        self.I_3_3_c1 = self.functionise_sympy([x_1, x_2], I_3_3_c1)
        self.I_3_4 = self.functionise_sympy([x_1, x_2], I_3_4_c1)

        I_4_right_integrand = (z_1-1)*(z_2-1) * (1 - epsilon * (z_1-z_2))**2
        I_4_left_integrand = (z_1-1)*(z_2-1) * (1 - epsilon * (z_2-z_1))**2

        I_4_1_left = sp.integrate(I_4_left_integrand, (z_1, x_1, z_2))
        I_4_1_right = sp.integrate(I_4_right_integrand, (z_1, z_2, 1))
        I_4_1 = sp.integrate(I_4_1_right + I_4_1_left, (z_2, x_2, 1))

        I_4_2_right = sp.integrate(I_4_right_integrand, (z_2, x_2, z_1-eps_inv))
        I_4_2 = sp.integrate(I_4_2_right, (z_1, x_2+eps_inv, 1))

        I_4_3_left = sp.integrate(I_4_left_integrand, (z_1, sp.Max(x_2-eps_inv, x_1), z_2-eps_inv))
        I_4_3 = sp.integrate(I_4_3_left, (z_2, sp.Max(x_2, x_1+eps_inv), 1))

        I_4_4_left = sp.integrate(I_4_left_integrand, (z_2, x_2, 1))
        I_4_4 = sp.integrate(I_4_4_left, (z_1, x_1, x_2-eps_inv))

        self.I_4_1 = self.functionise_sympy([x_1, x_2], I_4_1)
        self.I_4_2 = self.functionise_sympy([x_1, x_2], I_4_2)
        self.I_4_3 = self.functionise_sympy([x_1, x_2], I_4_3)
        self.I_4_4 = self.functionise_sympy([x_1, x_2], I_4_4)

        # x' < x - eps_inv
        A_k_I_1a = sp.integrate(x_2*(z_2-1)*(1-epsilon*(x_1-z_2))**2, (z_2, sp.Max(x_1-eps_inv, 0), x_1))
        A_k_I_1b = sp.integrate(x_2*(z_2-1)*(1-epsilon*(z_2-x_1))**2, (z_2, x_1, sp.Min(x_1+eps_inv, 1)))

        # x' > x + eps_inv
        A_k_I_2a = sp.integrate((x_2-1)*z_2*(1-epsilon*(x_1-z_2))**2, (z_2, sp.Max(x_1-eps_inv, 0), x_1))
        A_k_I_2b = sp.integrate((x_2-1)*z_2*(1-epsilon*(z_2-x_1))**2, (z_2, x_1, sp.Min(x_1+eps_inv, 1)))

        # x' < x, x' > x - eps_inv
        A_k_I_4a = sp.integrate((x_2-1)*z_2*(1-epsilon*(x_1-z_2))**2, (z_2, sp.Max(x_1-eps_inv, 0), x_2))
        A_k_I_4b = sp.integrate(x_2*(z_2-1)*(1-epsilon*(x_1-z_2))**2, (z_2, x_2, x_1))
        A_k_I_4c = A_k_I_1b

        # x' > x, x' < x + eps_inv
        A_k_I_5a = A_k_I_2a
        A_k_I_5b = sp.integrate((x_2-1)*z_2*(1-epsilon*(z_2-x_1))**2, (z_2, x_1, x_2))
        A_k_I_5c = sp.integrate(x_2*(z_2-1)*(1-epsilon*(z_2-x_1))**2, (z_2, x_2, sp.Min(x_1+eps_inv, 1)))

        self.A_k_I_1 = self.functionise_sympy([x_1, x_2], A_k_I_1a + A_k_I_1b)
        self.A_k_I_2 = self.functionise_sympy([x_1, x_2], A_k_I_2a + A_k_I_2b)
        self.A_k_I_3 = self.functionise_sympy([x_1, x_2], A_k_I_2a + A_k_I_1b)
        self.A_k_I_4 = self.functionise_sympy([x_1, x_2], A_k_I_4a + A_k_I_4b + A_k_I_4c)
        self.A_k_I_5 = self.functionise_sympy([x_1, x_2], A_k_I_5a + A_k_I_5b + A_k_I_5c)

    def I_1(self, x_1, x_2):
        eps_inv = 1./self.epsilon
        assert eps_inv < 1, 'We assume non-global support!'
        ret = self.I_1_1(x_1, x_2)
        if x_1 > eps_inv:
            ret -= self.I_1_2(x_1, x_2)
        if x_2 > eps_inv:
            ret -= self.I_1_3(x_1, x_2)
        if x_2 > x_1 + eps_inv:
            ret -= self.I_1_4(x_1, x_2)
        return ret

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

    def I_4(self, x_1, x_2):
        eps_inv = 1./self.epsilon
        if x_1 > x_2:
            raise Exception('We are assuming x_1 < x_2')
        ret = self.I_4_1(x_1, x_2)
        if x_2 < 1 - eps_inv:
            ret -= self.I_4_2(x_1, x_2)
        if x_1 < 1 - eps_inv:
            ret -= self.I_4_3(x_1, x_2)
        if x_1 < x_2 - eps_inv:
            ret -= self.I_4_4(x_1, x_2)
        return ret

    def kern(self, x_1, x_2):
        if x_1 > x_2:
            return self.kern(x_2, x_1)

        I_1 = self.I_1(x_1, x_2)
        I_2 = self.I_2(x_1, x_2)
        I_3 = self.I_3(x_1, x_2)
        I_4 = self.I_4(x_1, x_2)

        return (x_1-1)*(x_2-1)*I_1 + (x_1-1)*x_2*I_2 + x_1*(x_2-1)*I_3 + x_1*x_2*I_4

    def A_k(self, x_1, x_2):
        eps_inv = 1./self.epsilon
        if x_2 <= x_1 - eps_inv:
            return self.A_k_I_1(x_1, x_2)
        if x_2 >= x_1 + eps_inv:
            return self.A_k_I_2(x_1, x_2)
        if x_1 == x_2:
            return self.A_k_I_3(x_1, x_2)
        if x_1 - eps_inv < x_2 < x_1:
            return self.A_k_I_4(x_1, x_2)
        if x_1 < x_2 < x_1 + eps_inv:
            return self.A_k_I_5(x_1, x_2)

        raise Exception('Case appears to be missing! x_1 ={}, x_2 = {}'.format(x_1, x_2))

    def A_bar_k(self, x_1, x_2):
        return self.A_k(x_2, x_1)

    def A_A_bar_k(self, x_1, x_2):
        ret = np.maximum(1 - self.epsilon*(np.abs(x_1-x_2)), 0)
        return ret ** 2

    def __pairwiseify__(self, function):
        def __return(x_1, x_2, fun_args=None):
            return pairwise.apply(function, x_1, x_2)
        return __return

    def functionise_sympy(self, symbols, sympy_object):
        modules = [
            {'Min': min, 'Max': max},
            'numpy'
        ]
        return sp.lambdify(symbols, sympy_object, modules=modules)

    def __getitem__(self, item):
        if not type(item) is tuple:
            item = (item, )
        item = [i for i in item if i != ()]
        item = tuple(item)

        if item == () or item == (self.B,) or item == (self.B_bar,):
            return self.__pairwiseify__(self.kern)
        if item == (self.A,) or item == (self.A, self.B_bar):
            return self.__pairwiseify__(self.A_k)
        if item == (self.A_bar,) or item == (self.B, self.A_bar):
            return self.__pairwiseify__(self.A_bar_k)
        if item == (self.A, self.A_bar):
            return self.__pairwiseify__(self.A_A_bar_k)
        raise Exception('Operator not understood: {}'.format(item))


