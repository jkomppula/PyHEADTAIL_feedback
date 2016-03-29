import numpy as np
from scipy.constants import c, e
import scipy.integrate as integrate
import scipy.special as special
import sys
import itertools

def matrixGeneratorFactory(function,norm_range = None):
    if norm_range is None:
        norm_range = [-3, 3]
    norm_coeff, _ = integrate.quad(function, norm_range[0], norm_range[1], limit=1000)

    print 'Norm coeff: ' + str(norm_coeff) + ' range: ' + str(norm_range[0]) + ' - ' + str(norm_range[1])


    def generator(bin_set, bin_midpoints=None):
        if bin_midpoints is None:
            bin_midpoints = [(i+j)/2 for i, j in zip(bin_set, bin_set[1:])]

        matrix = np.identity(len(bin_midpoints))

        for i, midpoint in enumerate(bin_midpoints):
                for j in range(len(bin_midpoints)):
                    print 'Range: ' + str((bin_set[j]-midpoint)) + ' - ' + str((bin_set[j+1]-midpoint))

                    temp, _ = integrate.quad(function,(bin_set[j]-midpoint),(bin_set[j+1]-midpoint))
                    matrix[i][j] = temp/norm_coeff
        return matrix

    return generator


def phase_linearized_lowpass(f_cutoff):
    def f(dz):

        scaling = f_cutoff/c

        if dz == 0:
            return sys.float_info.max
        elif abs(scaling*dz)>10:
            return 0
        else:
            return special.k0(abs(scaling*dz))
    return f


def lowpass(f_cutoff):
    def f(dz):
        if dz < 0:
            return 0
        else:
            return np.exp(-1.*dz*f_cutoff/c)
    return f