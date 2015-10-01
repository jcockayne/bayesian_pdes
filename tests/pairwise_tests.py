__author__ = 'benorn'
import unittest
from bayesian_pdes import pairwise
import numpy as np
from timeit import timeit

def manual_pairwise(fun, A, B):
    ret = np.empty((A.shape[0], B.shape[0]))
    for i in range(A.shape[0]):
        for j in range(B.shape[0]):
            ret[i, j] = fun(A[i,:], B[j,:])
    return ret

class PairwiseTests(unittest.TestCase):
    def test_pairwise_1d(self):
        def __fun(a, b):
            assert a.shape == b.shape == (1,)
            return a * b

        A = np.array([[0., 1.]]).T
        B = np.array([[-1., 1.]]).T

        res = pairwise.apply(__fun, A, B)
        expected = np.array([[0, 0], [-1, 1]])
        np.testing.assert_array_equal(res, expected)

    def test_pairwise_2d(self):
        def __fun(a, b):
            assert a.shape == b.shape == (2,)
            return np.sum(a*b)

        A = np.array([[0., 1.], [1., 2.]])
        B = np.array([[1., 0.], [1., 2.]])

        res = pairwise.apply(__fun, A, B)
        expected = np.array([[0., 2.], [1., 5.]])
        np.testing.assert_array_equal(res, expected)

    def test_timing(self):
        def __fun(a, b):
            return np.sum(a*b)

        A,B = np.mgrid[0:1:100j, 0:1:100j]

        cython_timing = timeit(lambda: pairwise.apply(__fun, A, B), number=10)
        print cython_timing
        python_timing = timeit(lambda: manual_pairwise(__fun, A, B), number=10)
        print python_timing

        self.assertLess(cython_timing, python_timing)