import unittest
import numpy as np
from bayesian_pdes.util import bccb


def assert_exists_close(arr, in_arr, err_msg):
    assert len(arr) == len(in_arr), "Arrays are not of the same length"

    found = np.zeros(in_arr.shape, dtype=np.bool)
    for item in arr:
        delta = (in_arr[~found] - item).dot(in_arr[~found] - item)
        min_delta = np.argmin(delta)
        closest = in_arr[~found][min_delta]

        in_arr_ix = np.arange(len(in_arr))[~found][min_delta]

        np.testing.assert_approx_equal(item, closest, err_msg=err_msg)
        found[in_arr_ix] = True


def make_random_circulant(block_size, n_rows):
    if n_rows == 1:
        circ = np.random.normal(size=block_size)
        m = np.empty((block_size, block_size))
        for i in xrange(block_size):
            m[i] = np.roll(circ, i)
        return np.dot(m, m.T)
    matrices = [make_random_circulant(block_size, 1) for _ in xrange(n_rows)]
    ret = []
    for i in range(n_rows-1, -1, -1):
        ret.append(np.column_stack(matrices[i:]+matrices[:i]))
    return np.row_stack(ret)


class BCCBTest(unittest.TestCase):
    def test_eigs(self):
        for n_rows, block_size in zip([2,3,4,5], [2,3,4,5]):
            circ_mat = make_random_circulant(block_size, n_rows)

            eigs, _ = np.linalg.eig(circ_mat)
            eigs = np.sort(eigs)

            # for eac
            circ_eigs = np.sort(bccb.bccb_eigs(circ_mat[:,0], (block_size, n_rows)))
            norm_diff = np.linalg.norm(circ_eigs-eigs)
            assert_exists_close(eigs, circ_eigs, err_msg='Failed for n_rows={}, block_size={}, norm diff is {}'.format(n_rows, block_size, norm_diff))

    # todo: this is a pain because creating a PD circulant matrix is hard.
    def test_solve(self):
        for n_rows, block_size in zip([2,3,4,5], [2,3,4,5]):
            circ_mat = make_random_circulant(block_size, n_rows)
            b = np.random.normal(size=circ_mat.shape[0])

            x = np.linalg.solve(circ_mat, b)

            circ_x = bccb.bccb_solve(circ_mat[:,0], b, (block_size, n_rows))
            np.testing.assert_almost_equal(x, np.real(circ_x))


    # todo: this is a pain because creating a PD circulant matrix is hard.
    def test_solve_matrix(self):
        for n_rows, block_size in zip([2,3,4,5], [2,3,4,5]):
            circ_mat = make_random_circulant(block_size, n_rows)
            b = np.random.normal(size=(circ_mat.shape[0], 5))

            circ_x = bccb.bccb_solve(circ_mat[:,0], b, (block_size, n_rows))
            for ix in range(b.shape[1]):

                x = np.linalg.solve(circ_mat, b[:,ix])

                np.testing.assert_almost_equal(np.real(circ_x[:,ix]), x, err_msg='Failed for column {}'.format(ix))