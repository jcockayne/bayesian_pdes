cimport numpy as np
import numpy as np
#ctypedef np.float_t DTYPE_t

def apply(fun, np.ndarray[np.float_t, ndim=2] A, np.ndarray[np.float_t, ndim=2] B):
    cdef int i,j

    ret = np.empty((A.shape[0], B.shape[0]), dtype=np.float)
    for i in xrange(A.shape[0]):
        for j in xrange(B.shape[0]):
            ret[i,j] = fun(A[i,:], B[j,:])
    return ret