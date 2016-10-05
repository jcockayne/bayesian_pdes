import numpy as np
from scipy.sparse import linalg as splinalg
import abc


def factory(obj):
    if type(obj) is str:
        if obj == 'direct':
            return DirectInversion
        if obj == 'cg':
            return CGInversion
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


class CGInversion(Inverter):
    def __init__(self, mat, tol=1e-8):
        self.__mat__ = mat
        self.__tol__ = tol

    @staticmethod
    def apply_static(mat, rhs, tol):
        ret = np.empty((mat.shape[0], rhs.shape[1]))
        for ix in xrange(rhs.shape[1]):
            result, info = splinalg.cg(mat, rhs[:, ix], tol=tol)
            ret[:, ix] = result

        return ret

    def apply(self, rhs):
        return self.apply_static(self.__mat__, rhs, self.__tol__)

    def apply_left(self, lhs):
        return self.apply_static(self.__mat__.T, lhs.T, self.__tol__).T

