try:
    from autograd import numpy as np
except:
    import numpy as np
from scipy.sparse import linalg as splinalg
import abc


def factory(obj):
    if type(obj) is str:
        if obj == 'direct':
            return DirectInversion
        if obj == 'cg':
            return CGInversion
        if obj == 'np':
            return NPSolveInversion
    raise Exception('Inverter {} not understood'.format(obj))


class Inverter(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def apply(self, rhs):
        pass

    @abc.abstractmethod
    def apply_left(self, lhs):
        pass


class DirectInversion(Inverter):
    def __init__(self, mat):
        self.__inv__ = np.linalg.inv(mat)

    def apply(self, rhs):
        return np.dot(self.__inv__, rhs)

    def apply_left(self, lhs):
        return np.dot(lhs, self.__inv__)


class SolverInverter(Inverter):
    __metaclass__ = abc.ABCMeta

    def __init__(self, mat):
        self.__mat__ = mat

    @abc.abstractmethod
    def solve(self, mat, rhs):
        pass

    def apply_static(self, mat, rhs):
        ret = np.empty((mat.shape[0], rhs.shape[1]))
        for ix in xrange(rhs.shape[1]):
            result = self.solve(mat, rhs[:, ix])
            ret[:, ix] = result

        return ret

    def apply(self, rhs):
        return self.apply_static(self.__mat__, rhs)

    def apply_left(self, lhs):
        return self.apply_static(self.__mat__.T, lhs.T).T


class CGInversion(SolverInverter):
    def __init__(self, mat, tol=1e-8):
        super(CGInversion, self).__init__(mat)
        self.__tol__ = tol

    def solve(self, mat, rhs):
        result, info = splinalg.cg(mat, rhs, tol=self.__tol__)
        if info != 0:
            raise Exception("Inversion failed with nonzero exit code {}".format(info))
        return result


class NPSolveInversion(SolverInverter):
    def solve(self, mat, rhs):
        return np.linalg.solve(mat, rhs)
