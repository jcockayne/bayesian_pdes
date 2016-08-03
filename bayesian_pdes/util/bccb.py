import numpy as np
"""
Module for operations involving `block circulant circulant block' matrices.
"""

def __bccb_reshape__(vec, bccb_shape):
    """
    Reshape the vector to the format required by numpy's FFT
    :param vec: The vector
    :param bccb_shape: The shape, in the format (block size, number_of_blocks)
    :return: The reshaped vector
    """
    return vec.reshape(bccb_shape, order='F')


def __bccb_unreshape__(vec):
    return vec.ravel(order='F')


def bccb_eigs(a_1, bccb_shape, reshape=True):
    """
    Calculate the eigenvalues of a BCCB matrix A with first column a_1
    :param a_1: The first column of the BCCB matrix
    :param bccb_shape: The shape, in the format (block size, number_of_blocks)
    :param reshape: Whether the output should be returned as a column vector (default) or in the output format from
    FFT
    :return: The eigenvalues in the format specified by the 'reshape' param
    """
    A_1 = __bccb_reshape__(a_1, bccb_shape)
    lamb = np.fft.fft2(A_1)
    return __bccb_unreshape__(lamb) if reshape else lamb


def bccb_solve(a_1, b, bccb_shape, eigs=None):
    """
    Solve the linear system Ax = b for a BCCB A, with first column a_1
    :param a_1: The first column of the BCCB matrix
    :param b: right-hand-side of the system (or a matrix of many right-hand-sides
    :param bccb_shape: The shape, in the format (block size, number_of_blocks)
    :param eigs: Eigenvalues for the system as returned by bccb_eigs (ordered)
    :return: The solution to the linear system, or a matrix of solutions if b is a matrix.
    """
    if eigs is None:
        lamb = bccb_eigs(a_1, bccb_shape, reshape=False)
    else:
        lamb = eigs
        if np.ndim(lamb) == 1:
            lamb = __bccb_reshape__(lamb, bccb_shape)
    if np.ndim(b) == 1:
        B = __bccb_reshape__(b, bccb_shape)
        X = np.fft.ifft2(np.fft.fft2(B) / lamb)
        return __bccb_unreshape__(X)
    else:
        res = np.empty((a_1.shape[0], b.shape[1]))
        for ix in xrange(b.shape[1]):
            B = __bccb_reshape__(b[:, ix], bccb_shape)
            X = np.fft.ifft2(np.fft.fft2(B) / lamb)
            res[:, ix] = __bccb_unreshape__(X)
        return res
