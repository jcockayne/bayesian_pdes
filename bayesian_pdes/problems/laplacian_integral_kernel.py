import sympy as sp
from bayesian_pdes import pairwise


class LaplacianIntegralKernel(object):
    def __init__(self, epsilon):
        x, y = sp.symbols('x y')
        z = sp.Symbol('z')
        eps_inv = 1./epsilon

        def phi_L(m, n):
            return (1-epsilon*(m-n))**4 * (4*epsilon*(m-n) + 1)

        def phi_R(m, n):
            return phi_L(n, m)

        I1_1 = sp.integrate(phi_L(x, z)*phi_L(y, z), (z, sp.Max(y-eps_inv, 0), x))
        I1_2 = sp.integrate(phi_R(x, z)*phi_L(y, z), (z, sp.Max(x, y-eps_inv), sp.Min(y, x+eps_inv)))
        I1_3 = sp.integrate(phi_R(x, z)*phi_R(y, z), (z, y, sp.Min(x+eps_inv, 1)))

        self.I1_1 = self.functionise_sympy([x, y], I1_1)
        self.I1_2 = self.functionise_sympy([x, y], I1_2)
        self.I1_3 = self.functionise_sympy([x, y], I1_3)

        self.epsilon = epsilon
        self.eps_inv = eps_inv

        def lap_phi_L(m, n):
            return -20*epsilon**2*(1-epsilon*(m-n))**3 + 60*epsilon**3*(m-n)*(1-epsilon*(m-n))**2
        def lap_phi_R(m, n):
            return lap_phi_L(n, m)

        I2_1 = sp.integrate(lap_phi_L(x, z)*phi_L(y, z), (z, sp.Max(y-eps_inv, 0), x))
        I2_2 = sp.integrate(lap_phi_R(x, z)*phi_L(y, z), (z, sp.Max(x, y-eps_inv), sp.Min(y, x+eps_inv)))
        I2_3 = sp.integrate(lap_phi_R(x, z)*phi_R(y, z), (z, y, sp.Min(x+eps_inv, 1)))

        I3_1 = sp.integrate(phi_L(x, z)*lap_phi_L(y, z), (z, sp.Max(y-eps_inv, 0), x))
        I3_2 = sp.integrate(phi_R(x, z)*lap_phi_L(y, z), (z, sp.Max(x, y-eps_inv), sp.Min(y, x+eps_inv)))
        I3_3 = sp.integrate(phi_R(x, z)*lap_phi_R(y, z), (z, y, sp.Min(x+eps_inv, 1)))

        self.I2_1 = self.functionise_sympy([x, y], I2_1)
        self.I2_2 = self.functionise_sympy([x, y], I2_2)
        self.I2_3 = self.functionise_sympy([x, y], I2_3)

        self.I3_1 = self.functionise_sympy([x, y], I3_1)
        self.I3_2 = self.functionise_sympy([x, y], I3_2)
        self.I3_3 = self.functionise_sympy([x, y], I3_3)

        I4_1 = sp.integrate(lap_phi_L(x, z)*lap_phi_L(y, z), (z, sp.Max(y-eps_inv, 0), x))
        I4_2 = sp.integrate(lap_phi_R(x, z)*lap_phi_L(y, z), (z, sp.Max(x, y-eps_inv), sp.Min(y, x+eps_inv)))
        I4_3 = sp.integrate(lap_phi_R(x, z)*lap_phi_R(y, z), (z, y, sp.Min(x+eps_inv, 1)))
        self.I4_1 = self.functionise_sympy([x, y], I4_1)
        self.I4_2 = self.functionise_sympy([x, y], I4_2)
        self.I4_3 = self.functionise_sympy([x, y], I4_3)


        self.A = 'A'
        self.A_bar = 'A_bar'
        self.B = 'B'
        self.B_bar = 'B_bar'

        self.operators = [self.A, self.B]
        self.operators_bar = [self.A_bar, self.B_bar]

    def functionise_sympy(self, symbols, sympy_object):
        modules = [
            {'Min': min, 'Max': max},
            'numpy'
        ]
        return sp.lambdify(symbols, sympy_object, modules=modules)

    def kern(self, x, y):
        eps_inv = self.eps_inv
        if abs(y-x) >= 2*eps_inv:
            return 0.
        if x > y:
            return self.kern(y, x)
        ret = self.I1_2(x, y)
        if abs(y-x) < eps_inv:
            ret += self.I1_1(x, y) + self.I1_3(x, y)
        return ret

    def A_k(self, x, y):
        eps_inv = self.eps_inv
        if abs(y-x) >= 2*eps_inv:
            return 0.
        if x > y:
            return self.Abar_k(y, x)
        ret = 0
        if x != y:
            ret += self.I2_2(x, y)
        if y-x < eps_inv:
            ret += self.I2_1(x, y) + self.I2_3(x, y)
        return ret

    def Abar_k(self, x, y):
        eps_inv = self.eps_inv
        if abs(y-x) >= 2*eps_inv:
            return 0.
        if x > y:
            return self.A_k(y, x)
        ret = 0
        if x != y:
            ret += self.I3_2(x, y)
        if y-x < eps_inv:
            ret += self.I3_1(x, y) + self.I3_3(x, y)
        return ret

    def A_Abar_k(self, x, y):
        eps_inv = self.eps_inv
        if abs(y-x) >= 2*eps_inv:
            return 0.
        if x > y:
            return self.A_Abar_k(y, x)
        ret = 0
        if x != y:
            ret += self.I4_2(x, y)
        if y-x < eps_inv:
            ret += self.I4_1(x, y) + self.I4_3(x, y)
        return ret

    def __pairwiseify__(self, function):
        def __return(x_1, x_2, fun_args=None):
            return pairwise.apply(function, x_1, x_2)
        return __return

    def __getitem__(self, item):
        if not type(item) is tuple:
            item = (item, )

        if item == () or item == (self.B,) or item == (self.B_bar,) or item == (self.B, self.B_bar):
            return self.__pairwiseify__(self.kern)
        if item == (self.A,) or item == (self.A, self.B_bar):
            return self.__pairwiseify__(self.A_k)
        if item == (self.A_bar,) or item == (self.B, self.A_bar):
            return self.__pairwiseify__(self.Abar_k)
        if item == (self.A, self.A_bar):
            return self.__pairwiseify__(self.A_Abar_k)
        raise Exception('Operator not understood: {}'.format(item))