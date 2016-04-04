import numpy as np
from scipy.constants import c, e, pi
import scipy.integrate as integrate
import scipy.special as special
from collections import deque
import sys
import itertools
import math

class MatrixGenerator(object):
    def __init__(self,function,scaling = None, norm_type = None, norm_range = None):
        self.function = function
        self.norm_type = norm_type
        self.norm_range = norm_range
        self.scaling = scaling

        if self.scaling is None:
            self.scaling = 1

    def generate(self,bin_set, bin_midpoints=None):

        if bin_midpoints is None:
            bin_midpoints = [(i+j)/2 for i, j in zip(bin_set, bin_set[1:])]

        if self.norm_type == 'BunchLength':
            self.norm_coeff=bin_set[-1]-bin_set[0]
        elif self.norm_type == 'FixedLength':
            self.norm_coeff=self.norm_range[1]-self.norm_range[0]
        elif self.norm_type == 'BunchLengthInteg':
            self.norm_coeff, _ = integrate.quad(self.function, self.scaling*bin_set[0], self.scaling*bin_set[-1])
        elif self.norm_type == 'FixedLengthInteg':
            self.norm_coeff, _ = integrate.quad(self.function, self.scaling*self.norm_range[0], self.scaling*self.norm_range[1])
            print 'Norm coeff: ' + str(self.norm_coeff) + ' range: ' + str(self.norm_range[0]) + ' - ' + str(self.norm_range[1]) + ':' + str(self.function(self.norm_range[0])) + ' - ' + str(self.function(self.norm_range[1]))
        elif self.norm_type == 'MatrixSum':
            center_idx = math.floor(len(bin_midpoints)/2)
            self.norm_coeff, _ = integrate.quad(self.function,self.scaling*(bin_set[0]-bin_midpoints[center_idx]),self.scaling*(bin_set[-1]-bin_midpoints[center_idx]))
        else:
            self.norm_coeff = 1

        matrix = np.identity(len(bin_midpoints))

        for i, midpoint in enumerate(bin_midpoints):
                for j in range(len(bin_midpoints)):
                    temp, _ = integrate.quad(self.function,self.scaling*(bin_set[j]-midpoint),self.scaling*(bin_set[j+1]-midpoint))
                    matrix[i][j] = temp/self.norm_coeff
        return matrix

class LinearProcessor(object):
    """ General class for linear signal processing. The emitted signal is a dot product of
    a transfer matrix and the incoming signal. The transfer matrix is produced from response function and
    (possible non uniform) z_bin_set by using MatrixGenerator class"""

    def __init__(self,response_function, scaling=None, norm_type=None, norm_range=None):
        self.response_function = response_function
        self.scaling = scaling
        self.norm_type = norm_type
        self.norm_range = norm_range
        self.matrix_generator = MatrixGenerator(self.response_function, self.scaling, self.norm_type, self.norm_range)

        self.z_bin_set = [sys.float_info.max]

        self.matrix = None

    def process(self,signal,slice_set):

        if self.check_bin_set(slice_set.z_bins):
            #print 'Matrix recalculated'
            self.z_bin_set = slice_set.z_bins
            self.matrix = self.matrix_generator.generate(slice_set.z_bins,slice_set.mean_z)

        return np.dot(self.matrix,signal)

    def check_bin_set(self,z_bin_set):
        """Check if bin_set has been changed more than given limit"""
        changed = False

        for old, new in itertools.izip(self.z_bin_set,z_bin_set):
            if old != new:
            #if new/old<1.0001 and new/old>0.9999:
                #print str(old) + ' != ' + str(new) + ': ',
                changed = True
                break

        return changed

    def print_matrix(self):
        for row in self.matrix:
            print "[",
            for element in row:
                print "{:6.3f}".format(element),
            print "]"


class IdealSlice(object):
    def process(self,signal,slice_set):
        return signal


class IdealBunch(LinearProcessor):
    """An ideal bunch feedback corresponds to an uniform matrix in the linear signal processor"""
    def __init__(self,norm_type = 'MatrixSum', norm_range = None):
        self.norm_type = norm_type
        self.norm_range = norm_range
        self.scaling = 1
        super(self.__class__, self).__init__(self.response_function, self.scaling, self.norm_type, self.norm_range)

    def response_function(self,x):
        return 1


class PhaseLinearizedLowpass(LinearProcessor,):
    def __init__(self, f_cutoff, norm_type = 'BunchLengthInteg', norm_range = None):
        self.norm_type = norm_type
        self.norm_range = norm_range
        self.scaling = f_cutoff/c
        self.norm_range_coeff = 10
        self.norm_range = [-1.0 * self.norm_range_coeff / self.scaling, self.norm_range_coeff / self.scaling]
        super(self.__class__, self).__init__(self.f, self.scaling, self.norm_type, self.norm_range)

    def f(self,x):
        if x == 0:
            return 0
        else:
            return special.k0(abs(x))

# TODO: Check vector sum of complex numbers
class Register(object):
    """Holds signal values . Returns xxx. Phase shift is taken into account by """
    def __init__(self,length,turn_phase_shift, avg_length=1, position_phase_shift = 0):
        self.length = length
        self.turn_phase_shift = turn_phase_shift
        self.position_phase_shift = position_phase_shift
        self.avg_length = avg_length
        self.register = deque()

    def process(self,signal,slice_set):

        self.register.append(signal)

        if len(self.register) > self.length:
            self.register.popleft()

        if(self.avg_length>1):

            output = np.zeros(len(self.register[0]))
            if len(self.register) < self.avg_length:
                to = len(self.register)
            else:
                to = self.avg_length

            for i in range(to):
                output += 2.0*math.cos(((1-len(self.register))+i)*self.turn_phase_shift+self.position_phase_shift)*self.register[i]/self.avg_length
            return output
        else:
            return math.cos((1-len(self.register))*self.turn_phase_shift+self.position_phase_shift)*self.register[0]


# def lowpass(f_cutoff):
#     def f(dz):
#         if dz < 0:
#             return 0
#         else:
#             return np.exp(-1.*dz*f_cutoff/c)
#     return f

class ChargeWeighting:
    """ Weights value of each slice by its charge."""
    def process(self,signal,slice_set):
        n_macroparticles = np.sum(slice_set.n_macroparticles_per_slice)
        return np.array([signal*weight for signal, weight in itertools.izip(signal, slice_set.n_macroparticles_per_slice)])*slice_set.n_slices/n_macroparticles