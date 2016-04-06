import numpy as np
from scipy.constants import c, e, pi
import scipy.integrate as integrate
import scipy.special as special
from collections import deque
import sys
import itertools
import math

#TODO: add noise generator

class LinearProcessor(object):
    """ General class for linear signal processing. The signal is processed by calculating a dot product of a transfer matrix and a signal. The transfer matrix is produced
    from response function and (possible non uniform) z_bin_set by using generate_matrix function.
    """

    def __init__(self,response_function, scaling=None, norm_type=None, norm_range=None):
        """ General constructor to create a LinearProcessor.

        :param response_function: Impulse response function of the processor
        :param scaling: Because integration by substitution doesn't work with np.quad (see quad_problem.ipynbl), it
        must be done by scaling integral limits. This parameter is a linear scaling coefficient of the integral limits.
        An ugly way which must be fixed.
        :param norm_type: Describes a normalization method for the transfer matrix
        :param norm_range: Normalization range in the case of norm_type='FixedLength'
        """

        self.response_function = response_function
        self.scaling = scaling
        if self.scaling is None:
            self.scaling = 1
        self.norm_type = norm_type
        self.norm_range = norm_range

        self.z_bin_set = [sys.float_info.max]
        self.matrix = None

    def process(self,signal,slice_set, *args):
        """ Processes the signal. Recalculates the transfer matrix if the bin set is changed."""

        #TODO: This check should be optional
        if self.check_bin_set(slice_set.z_bins):
            self.z_bin_set = slice_set.z_bins
            self.matrix = self.generate_matrix(slice_set.z_bins,slice_set.mean_z)

        return np.dot(self.matrix,signal)

    def check_bin_set(self,z_bin_set):
        """Checks if bin_set has been changed."""
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

    def generate_matrix(self,bin_set, bin_midpoints=None):
        if bin_midpoints is None:
            bin_midpoints = [(i+j)/2 for i, j in zip(bin_set, bin_set[1:])]

        # TODO: Rethink these
        if self.norm_type == 'BunchLength':
            self.norm_coeff=bin_set[-1]-bin_set[0]
        elif self.norm_type == 'FixedLength':
            self.norm_coeff=self.norm_range[1]-self.norm_range[0]
        elif self.norm_type == 'BunchLengthInteg':
            self.norm_coeff, _ = integrate.quad(self.response_function, self.scaling*bin_set[0], self.scaling*bin_set[-1])
        elif self.norm_type == 'FixedLengthInteg':
            self.norm_coeff, _ = integrate.quad(self.response_function, self.scaling*self.norm_range[0], self.scaling*self.norm_range[1])
            print 'Norm coeff: ' + str(self.norm_coeff) + ' range: ' + str(self.norm_range[0]) + ' - ' + str(self.norm_range[1]) + ':' + str(self.response_function(self.norm_range[0])) + ' - ' + str(self.response_function(self.norm_range[1]))
        elif self.norm_type == 'MatrixSum':
            center_idx = math.floor(len(bin_midpoints)/2)
            self.norm_coeff, _ = integrate.quad(self.response_function,self.scaling*(bin_set[0]-bin_midpoints[center_idx]),self.scaling*(bin_set[-1]-bin_midpoints[center_idx]))
        else:
            self.norm_coeff = 1

        matrix = np.identity(len(bin_midpoints))

        for i, midpoint in enumerate(bin_midpoints):
                for j in range(len(bin_midpoints)):
                    temp, _ = integrate.quad(self.response_function,self.scaling*(bin_set[j]-midpoint),self.scaling*(bin_set[j+1]-midpoint))
                    matrix[i][j] = temp/self.norm_coeff
        return matrix

class Averager(LinearProcessor):
    """An ideal bunch feedback corresponds to an uniform matrix in the linear signal processor"""
    def __init__(self,norm_type = 'MatrixSum', norm_range = None):
        self.norm_type = norm_type
        self.norm_range = norm_range
        self.scaling = 1
        super(self.__class__, self).__init__(self.response_function, self.scaling, self.norm_type, self.norm_range)

    def response_function(self,x):
        return 1


class PhaseLinearizedLowpass(LinearProcessor):
    def __init__(self, f_cutoff, norm_type = 'MatrixSum', norm_range = None):
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

class LowpassFilter(LinearProcessor):
    def __init__(self, f_cutoff, norm_type = 'MatrixSum', norm_range = None):
        self.norm_type = norm_type
        self.norm_range = norm_range
        self.scaling = f_cutoff/c
        self.norm_range_coeff = 10
        self.norm_range = [-1.0 * self.norm_range_coeff / self.scaling, self.norm_range_coeff / self.scaling]
        super(self.__class__, self).__init__(self.f, self.scaling, self.norm_type, self.norm_range)

    def f(self,x):
        if x<0:
            return 0
        else:
            return math.exp(-1.*x)


class Bypass(object):
    def process(self,signal, *args):
        return signal


# def lowpass(f_cutoff):
#     def f(dz):
#         if dz < 0:
#             return 0
#         else:
#             return np.exp(-1.*dz*f_cutoff/c)
#     return f

# To
class Weighter(object):
    def __init__(self, weight_property, weight_function, weight_normalization):
        self.weight_property = weight_property
        self.weight_function = weight_function
        self.weight_normalization = weight_normalization
    """ Weights value of each slice by its charge."""
    def process(self,signal,slice_set):
        weight = np.array([])
        if self.weight_property == 'x':
            weight = np.array(slice_set.mean_x)
        elif self.weight_property == 'y':
            weight = np.array(slice_set.mean_y)
        elif self.weight_property == 'z':
            weight = np.array(slice_set.mean_z)
        elif self.weight_property == 'xp':
            weight = np.array(slice_set.mean_xp)
        elif self.weight_property == 'yp':
            weight = np.array(slice_set.mean_yp)
        elif self.weight_property == 'dp':
            weight = np.array(slice_set.mean_dp)
        elif self.weight_property == 'sigma_x':
            weight = np.array(slice_set.sigma_x)
        elif self.weight_property == 'sigma_y':
            weight = np.array(slice_set.sigma_y)
        elif self.weight_property == 'sigma_z':
            weight = np.array(slice_set.sigma_z)
        elif self.weight_property == 'sigma_xp':
            weight = np.array(slice_set.sigma_xp)
        elif self.weight_property == 'sigma_yp':
            weight = np.array(slice_set.sigma_yp)
        elif self.weight_property == 'sigma_dp':
            weight = np.array(slice_set.sigma_dp)
        elif self.weight_property == 'epsn_x':
            weight = np.array(slice_set.epsn_x)
        elif self.weight_property == 'epsn_y':
            weight = np.array(slice_set.epsn_y)
        elif self.weight_property == 'epsn_z':
            weight = np.array(slice_set.epsn_z)
        elif self.weight_property == 'charge':
            weight = np.array(slice_set.n_macroparticles_per_slice)
        elif self.weight_property == 'bin_length':
            bin_set = slice_set.z_bins
            weight = [(j-i) for i, j in zip(bin_set, bin_set[1:])]
        elif self.weight_property == 'bin_midpoint':
            bin_set = slice_set.z_bins
            weight = [(i+j)/2 for i, j in zip(bin_set, bin_set[1:])]

        weight = self.weight_function(weight)

        norm_coeff = 1.

        if self.weight_normalization == 'total_weight':
            norm_coeff = float(np.sum(weight))
        elif self.weight_normalization == 'average_weight':
            norm_coeff = float(np.sum(weight))/float(len(weight))
        elif self.weight_normalization == 'maximum_value':
            norm_coeff = float(min(weight))
        elif self.weight_normalization == 'minimum_value':
            norm_coeff = float(max(weight))

        return signal*weight/norm_coeff

class ChargeWeighter(Weighter):
    def __init__(self):
        super(self.__class__, self).__init__('charge', self.weight_function, 'average_weight')
    def weight_function(self,weight):
        return weight


# TODO: Initiliaze register
# TODO: Do a general register object with a changeable sum calculator
class Register(object):
    """Holds signal values . Returns xxx. Phase shift is taken into account by """
    def __init__(self,length,phase_shift_per_turn, avg_length=1, position_phase_angle = 0):
        self.length = length
        self.phase_shift_per_turn = phase_shift_per_turn
        self.position_phase_angle = position_phase_angle
        self.avg_length = avg_length
        self.register = deque()

    def process(self,signal, *args):

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
                output += 2.0*math.cos(((1-len(self.register))+i)*self.phase_shift_per_turn+self.position_phase_angle)*self.register[i]/self.avg_length
            return output
        else:
            return math.cos((1-len(self.register))*self.phase_shift_per_turn+self.position_phase_angle)*self.register[0]

    def read_signal(self,reader_phase_angle):
        #print 'Reader angle: ' + str(reader_phase_angle) + ' Register angle: ' + str(self.position_phase_angle)

        delta_Phi = self.position_phase_angle - reader_phase_angle

        if delta_Phi > 0:
            delta_Phi -= 2*pi

        if(self.avg_length>1):

            output = np.zeros(len(self.register[0]))
            if len(self.register) < self.avg_length:
                to = len(self.register)
            else:
                to = self.avg_length

            for i in range(to):
                output += 2.0*math.cos(((1-len(self.register))+i)*self.phase_shift_per_turn+delta_Phi)*self.register[i]/self.avg_length
            return output
        else:
            return math.cos((1-len(self.register))*self.phase_shift_per_turn+delta_Phi)*self.register[0]