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


def block_diag(arrs):
    arrs = [np.atleast_2d(a) for a in arrs]

    bad_args = [k for k in range(len(arrs)) if arrs[k].ndim > 2]
    if bad_args:
        raise ValueError("arguments in the following positions have dimension "
                            "greater than 2: %s" % bad_args)

    rows = []
    for ix, a in enumerate(arrs):
        left_shape = sum(a.shape[0] for a in arrs[:ix])
        right_shape = sum(a.shape[0] for a in arrs[ix+1:])
        left_mat = np.zeros((a.shape[0], left_shape))
        right_mat = np.zeros((a.shape[0], right_shape))
        rows.append(np.column_stack([left_mat, a, right_mat]))

    return np.row_stack(rows)