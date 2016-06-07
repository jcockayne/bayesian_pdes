import unittest
from bayesian_pdes.util.linalg import schur
import numpy as np


class LinalgTest(unittest.TestCase):
    def test_simple(self):
        A = np.eye(5)
        U = 0*A
        C = A

        mat = np.r_[np.c_[A, U], np.c_[U, C]]

        np.testing.assert_almost_equal(schur(A, U, C, C), mat)

    def test_complex(self):
        mat = np.random.normal(size=(1000,1000))
        mat = mat.dot(mat.T)

        p1 = 800

        tl = mat[:p1, :p1]
        br = mat[p1:, p1:]
        tr = mat[:p1, p1:]

        tl_inv = np.linalg.inv(tl)

        br_inv = np.linalg.inv(br)
        schur_inv = schur(tl_inv, tr, br, br_inv)
        np_inv = np.linalg.inv(mat)
        np.testing.assert_almost_equal(schur_inv, np_inv)