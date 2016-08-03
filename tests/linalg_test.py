import unittest
from bayesian_pdes.util.linalg import schur
from bayesian_pdes.util import linalg
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

    def test_block_diag(self):
        mat1 = np.random.normal(size=(4,4))
        mat2 = np.random.normal(size=(2,2))
        mat3 = np.random.normal(size=(5,5))
        mats = [mat1, mat2, mat3]

        diag = linalg.block_diag(mats)
        ix_i = 0
        for i, m1 in enumerate(mats):
            ix_j = 0
            for j, m2 in enumerate(mats):
                m = diag[ix_i:ix_i+m1.shape[0], ix_j:ix_j+m2.shape[1]]
                if i != j:
                    np.testing.assert_array_equal(m, np.zeros_like(m))
                else:
                    np.testing.assert_array_equal(m, m1)
                ix_j += m2.shape[1]
            ix_i += m1.shape[0]
