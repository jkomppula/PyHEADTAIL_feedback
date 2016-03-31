import numpy as np
from scipy.constants import c, e
import scipy.integrate as integrate
import scipy.special as special
import sys
import itertools




def matrixGeneratorFactory(function,norm_range = None,scaling = None):
    if norm_range is None:
        norm_range = [-100, 100]
    if scaling is None:
        scaling = 1
    data = integrate.quad(function, scaling*norm_range[0], scaling*norm_range[1])
    #, limit=100000, epsrel=1.49e-14
    print 'Norm coeff: ' + str(data) + ' range: ' + str(norm_range[0]) + ' - ' + str(norm_range[1]) + ':' + str(function(norm_range[0])) + ' - ' + str(function(norm_range[1]))
    norm_coeff = data[0]

    def generator(bin_set, bin_midpoints=None):
        if bin_midpoints is None:
            bin_midpoints = [(i+j)/2 for i, j in zip(bin_set, bin_set[1:])]

        matrix = np.identity(len(bin_midpoints))

        for i, midpoint in enumerate(bin_midpoints):
                for j in range(len(bin_midpoints)):
                    temp, _ = integrate.quad(function,scaling*(bin_set[j]-midpoint),scaling*(bin_set[j+1]-midpoint))
                    matrix[i][j] = temp/norm_coeff
        return matrix

    return generator

class transfer_function(object):
    def __init__(self,distribution,norm_range,scaling):
        self.distribution = distribution
        self.norm_range = norm_range
        self.scaling = scaling
        self.matrixGenerator = matrixGeneratorFactory(self.distribution,self.norm_range,self.scaling)

    def matrix(self,slice_set):
        return self.matrixGenerator(slice_set.z_bins,slice_set.mean_z)

class phase_linearized_lowpass(transfer_function):
    def __init__(self, f_cutoff):

        self.scaling = f_cutoff/c
        self.norm_range_coeff = 10
        self.norm_range = [-1.0 * self.norm_range_coeff / self.scaling, self.norm_range_coeff / self.scaling]

        super(self.__class__, self).__init__(self.f,self.norm_range,self.scaling)

    def f(self,x):
        if x == 0:
            return 0
        else:
            return special.k0(abs(x))


class ideal_slice(object):

    def matrix(self,slice_set,*arg):
        matrix = np.identity(slice_set.n_slices)
        return matrix

class ideal_bunch(object):

    def matrix(self,slice_set,*arg):
        matrix = np.identity(slice_set.n_slices)
        matrix.fill(1.0/slice_set.n_slices)
        return matrix

def lowpass(f_cutoff):
    def f(dz):
        if dz < 0:
            return 0
        else:
            return np.exp(-1.*dz*f_cutoff/c)
    return f