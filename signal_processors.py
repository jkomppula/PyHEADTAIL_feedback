import numpy as np
from scipy.constants import c, e
import scipy.integrate as integrate
import scipy.special as special
import sys
import itertools




class MatrixGenerator(object):
    def __init__(self,function,norm_range = None,scaling = None):
        self.function = function
        self.norm_range = norm_range
        self.scaling = scaling

        if self.scaling is None:
            self.scaling = 1

        if self.norm_range is None:
            self.norm_coeff = 0
        else:
            data = integrate.quad(self.function, self.scaling*self.norm_range[0], self.scaling*self.norm_range[1])
            #, limit=100000, epsrel=1.49e-14
            print 'Norm coeff: ' + str(data) + ' range: ' + str(norm_range[0]) + ' - ' + str(norm_range[1]) + ':' + str(function(norm_range[0])) + ' - ' + str(function(norm_range[1]))
            self.norm_coeff = data[0]

    def generate(self,bin_set, bin_midpoints=None):

        if bin_midpoints is None:
            bin_midpoints = [(i+j)/2 for i, j in zip(bin_set, bin_set[1:])]

        if self.norm_coeff == 0:
            self.norm_coeff=bin_set[-1]-bin_set[0]
            #print str(bin_set[-1]) + ' - ' + str(bin_set[1]) + ' = ' + str(self.norm_coeff)

        matrix = np.identity(len(bin_midpoints))

        for i, midpoint in enumerate(bin_midpoints):
                for j in range(len(bin_midpoints)):
                    temp, _ = integrate.quad(self.function,self.scaling*(bin_set[j]-midpoint),self.scaling*(bin_set[j+1]-midpoint))
                    matrix[i][j] = temp/self.norm_coeff
        return matrix

class LinearProcessor(object):
    def __init__(self,response_function,norm_range,scaling):
        self.response_function = response_function
        self.norm_range = norm_range
        self.scaling = scaling

        self.matrix_generator = MatrixGenerator(self.response_function,self.norm_range,self.scaling)

        self.z_bin_set = [sys.float_info.max]

        self.matrix = None

    def process(self,signal,slice_set):

        if self.check_bin_set(slice_set.z_bins):
            self.z_bin_set = slice_set.z_bins
            self.matrix = self.matrix_generator.generate(slice_set.z_bins,slice_set.mean_z)

        return np.dot(self.matrix,signal)

    def check_bin_set(self,z_bin_set):
        changed = False

        for old, new in itertools.izip(self.z_bin_set,z_bin_set):
            if old != new:
                changed = True
                break

        return changed

    def print_matrix(self):
        for row in self.matrix:
            print "[",
            for element in row:
                print "{:6.3f}".format(element),
            print "]"



class PhaseLinearizedLowpass(LinearProcessor):
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

class Register:
    def __init__(self,size,phase_shift):
        self.size = size

    def process(self,signal,slice_set):
        return signal

class IdealSlice(object):
    def process(self,signal,slice_set):
        return signal

class IdealBunch(LinearProcessor):
    def __init__(self):
        super(self.__class__, self).__init__(self.response_function,norm_range=None,scaling=None)

    def response_function(self,x):
        return 1

# def lowpass(f_cutoff):
#     def f(dz):
#         if dz < 0:
#             return 0
#         else:
#             return np.exp(-1.*dz*f_cutoff/c)
#     return f

class ChargeWeighting:
    def process(self,signal,slice_set):
        n_macroparticles = np.sum(slice_set.n_macroparticles_per_slice)
        return np.array([signal*weight for signal, weight in itertools.izip(signal, slice_set.n_macroparticles_per_slice)])*slice_set.n_slices/n_macroparticles