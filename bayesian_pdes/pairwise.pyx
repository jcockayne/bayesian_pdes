cimport numpy as np
import numpy as np
#ctypedef np.float_t DTYPE_t

def apply(fun, np.ndarray[np.float_t, ndim=2] A, np.ndarray[np.float_t, ndim=2] B, np.ndarray[np.float_t, ndim=1] extra=None):
    cdef int i,j
    try:
        ret = np.empty((A.shape[0], B.shape[0]), dtype=np.float)
    except Exception as ex:
        raise Exception('Failed to allocate ({},{}) matrix: {}'.format(A.shape[0], B.shape[0], ex))
    for i in xrange(A.shape[0]):
        for j in xrange(B.shape[0]):
            ret[i,j] = fun(A[i,:], B[j,:]) if extra is None else fun(A[i,:], B[j,:], extra)
    return ret