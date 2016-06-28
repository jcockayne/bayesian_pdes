from autograd import numpy as np


def woodbury(A_inv, U, C):
    premult = np.dot(A_inv, U)
    postmult = premult.T
    interior = np.dot(U.T, premult) - C
    interior_inv = np.linalg.inv(interior)

    return A_inv - np.dot(premult, np.dot(interior_inv, postmult))


def schur(A_inv, U, C, C_inv):
    """
    Schur complement inverse for symmetric matrices of the form:
    [[A,   U],
     [U.T, C]]
    U represents the top-right corner
    """
    top_left = woodbury(A_inv, U, C)
    #top_left = np.linalg.inv(np.linalg.inv(A_inv) - np.dot(U, np.dot(C_inv, U.T)))
    premult = np.dot(C_inv, U.T)
    bottom_left = np.dot(premult, top_left)
    top_right = bottom_left.T
    bottom_right = C_inv + np.dot(bottom_left, premult.T)

    return np.concatenate([np.concatenate([top_left, -top_right], axis=1),
                           np.concatenate([-bottom_left, bottom_right], axis=1)])