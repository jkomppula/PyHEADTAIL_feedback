import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def cython_dot_p(double[:, ::1] matrix not None, double[::1] vector not None):

    cdef np.intp_t i, j, n_samples
    n_samples = matrix.shape[0]
    cdef double[::1] D = np.zeros(n_samples)
    
    for i in range(n_samples):
        for j in range(n_samples):
            D[i] += matrix[i,j]* vector[j]
            
    return D