#TODO: maybe this could be simplified/avoided by using cimport scipy.linalg.cython_blas

import numpy as np
cimport numpy as np
cimport cython

""" The functions in this file have been written, because the dot product function of NumPy slowed down PyHEADTAIL
    simulation in the CERN batch system by a factor of two or more. The only working solution which was found was to
    write a new function for matrix product in Cython.
"""

@cython.boundscheck(False)
@cython.wraparound(False)

def cython_circular_convolution(double[::1] signal not None, double[::1] impulse not None, int zero_bin):

    cdef np.intp_t i, j, dim_0, dim_1
    cdef np.intp_t source_index
    cdef np.float_t temp_value
    cdef np.intp_t signal_length = len(signal)
    cdef np.intp_t impulse_length = len(impulse)
    cdef double[::1] output = np.zeros(signal_length)

    for i in range(signal_length):
        temp_value = 0
        source_index = (i - zero_bin + signal_length)
        if source_index > (signal_length - 1):
            source_index = source_index - signal_length
        for j in range(impulse_length):
#            source_index = (i - zero_bin + j + signal_length)
            if (source_index == signal_length):
                source_index = 0

            temp_value += signal[signal_length - source_index -1] * impulse[j]
            source_index = source_index + 1

        output[signal_length - i -1] = temp_value


    return output

# def cython_matrix_product(double[:, ::1] matrix not None, double[::1] vector not None):
#
#    cdef np.intp_t i, j, dim_0, dim_1
#    dim_0 = matrix.shape[0]
#    dim_1 = matrix.shape[1]
#    cdef double[::1] D = np.zeros(dim_0)
#
#    for i in range(dim_0):
#        for j in range(dim_1):
#            D[i] += matrix[i,j]* vector[j]
#
#    return D
