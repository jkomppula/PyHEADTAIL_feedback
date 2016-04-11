import numpy as np
from scipy.constants import c, e, pi
import scipy.integrate as integrate
import scipy.special as special
from collections import deque
import sys
import itertools
import math

# TODO: add noise generator
# TODO: add realistic pickup
# TODO: add phase shifter
# TODO: alternative implementation by using convolution?


class GenericProcessor(object):
    def __init__(self):
        a=1


class NoiseGenerator(GenericProcessor):
    def __init__(self,RMS_noise_level,reference_level = 'absolute_signal', distribution = 'normal'):

        self.RMS_noise_level = RMS_noise_level
        self.reference_level = reference_level
        self.distribution = distribution

        super(self.__class__, self).__init__()

    def process(self,signal,slice_set):

        randoms = np.zeros(len(signal))

        if self.distribution == "normal" or self.distribution is None:
            randoms = np.random.randn(len(signal))
        elif self.distribution == "uniform":
            randoms = 1./0.577263*(-1.+2.*np.random.rand(len(signal)))



        if self.reference_level == 'absolute':
            signal = signal + self.RMS_noise_level*randoms
        elif self.reference_level == 'maximum':
            signal = self.RMS_noise_level*np.max(signal)*randoms*signal
        elif self.reference_level == 'relative':
            signal = self.RMS_noise_level*randoms*signal


        return signal


class PickUp(GenericProcessor):

    def __init__(self,RMS_noise_level,f_cutoff):
        self.RMS_noise_level = RMS_noise_level
        self.f_cutoff = f_cutoff

        self.noise_generator = NoiseGenerator(self.RMS_noise_level)
        self.filter = LowpassFilter(self.f_cutoff)
        self.charge_weighter = ChargeWeighter()


    def process(self,signal,slice_set):
        signal = self.charge_weighter.process(signal,slice_set)

        plate_1 = 1.*self.noise_generator.process(signal,slice_set)
        plate_1 = self.filter.process(plate_1,slice_set)
        plate_1 += 1.

        plate_2 = -1.*self.noise_generator.process(signal,slice_set)
        plate_2 = self.filter.process(plate_2,slice_set)
        plate_2 += 1

        return (plate_1-plate_2)/(plate_1+plate_2)





class LinearProcessor(object):
    """ General class for linear signal processing. The signal is processed by calculating a dot product of a transfer matrix and a signal. The transfer matrix is produced
    from response function and (possible non uniform) z_bin_set by using generate_matrix function.
    """

    def __init__(self,response_function, scaling=1., norm_type=None, norm_range=None):
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
            bin_midpoints = [(i+j)/2. for i, j in zip(bin_set, bin_set[1:])]

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
        super(self.__class__, self).__init__(self.response_function, 1, norm_type, norm_range)

    def response_function(self,x):
        return 1


class PhaseLinearizedLowpass(LinearProcessor):
    def __init__(self, f_cutoff, norm_type = 'MatrixSum', norm_range = None):
        scaling = f_cutoff/c
        self.norm_range_coeff = 10
        self.norm_range = [-1.0 * self.norm_range_coeff / scaling, self.norm_range_coeff / scaling]
        super(self.__class__, self).__init__(self.f, scaling, norm_type, norm_range)

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


class Weighter(object):
    """ A general class for signal weighing. A seed for the weight is a property of slices (weight_property).
        The weight is calculated by giving the seed to weight_function() as a parameter. A parameter
        weight_normalization determines which part of the weight is normalized to be one.
    """

    def __init__(self, weight_property, weight_function, weight_normalization = None):
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
        elif self.weight_property == 'signal':
            weight = np.array(signal)

        weight = self.weight_function(weight)

        if self.weight_normalization is None:
            norm_coeff =1
        elif self.weight_normalization == 'total_weight':
            norm_coeff = float(np.sum(weight))
        elif self.weight_normalization == 'average_weight':
            norm_coeff = float(np.sum(weight))/float(len(weight))
        elif self.weight_normalization == 'maximum_weight':
            norm_coeff = float(max(weight))
        elif self.weight_normalization == 'minimum_weight':
            norm_coeff = float(min(weight))

        return signal*weight/norm_coeff


class ChargeWeighter(Weighter):
    def __init__(self):
        super(self.__class__, self).__init__('charge', self.weight_function, 'average_weight')

    def weight_function(self,weight):
        return weight

class FermiDiracInverseWeighter(Weighter):
    def __init__(self,bunch_length,bunch_decay_length,maximum_weight = 10):
        self.bunch_length = bunch_length
        self.bunch_decay_length = bunch_decay_length
        self.maximum_weight=maximum_weight
        super(self.__class__, self).__init__('bin_midpoint', self.weight_function, 'minimum_weight')

    def weight_function(self,weight):
        weight = np.exp((np.absolute(weight)-self.bunch_length/2.)/float(self.bunch_decay_length))+ 1.
        weight = np.clip(weight,1.,self.maximum_weight)
        return weight




# TODO: Initiliaze register
# TODO: Rethink weighing
class Register(object):
    """ A general class for a signal register. A signal is stored to the register, when the function process() is
        called. Depending on the avg_length parameter, a return value of the process() function is and an averaged
        value of the stored signals.
        A effect of a betatron shift between turns and between the register and the reader is taken into
        account by calculating a weight for the register value with phase_weight_function(). Total phase differences are
        calculated with delta_phi_calculator. The register can be also ridden without changing it by calling read_signal.
        In this case a relative betatron phase angle of the reader must be given as a parameter.
    """

    def __init__(self,phase_weight_function, delta_phi_calculator, phase_shift_per_turn,delay, avg_length, position_phase_angle, n_slices):
        self.phase_weight_function = phase_weight_function
        self.delta_phi_calculator = delta_phi_calculator
        self.phase_shift_per_turn = phase_shift_per_turn
        self.delay = delay
        self.position_phase_angle = position_phase_angle
        self.avg_length = avg_length
        self.n_slices = n_slices

        self.max_reg_length = self.delay+self.avg_length
        self.register = deque()

    def process(self,signal, *args):

        self.register.append(signal)

        if len(self.register) > self.max_reg_length:
            self.register.popleft()

        return self.read_signal(None)

    def read_signal(self,reader_phase_angle):

        if len(self.register) == 0:
            if self.n_slices is not None:
                self.register.append(np.zeros(self.n_slices))

        if reader_phase_angle is None:
            delta_Phi = 0
        else:
            delta_Phi = self.delta_phi_calculator(self.position_phase_angle,reader_phase_angle)

        turns_to_read = min(self.avg_length,len(self.register))

        if(turns_to_read>1):
            output = np.zeros(len(self.register[0]))
            for i in range(turns_to_read):
                n_delay = 1-len(self.register)+i
                output += self.phase_weight_function(n_delay,self.phase_shift_per_turn,delta_Phi)*self.register[i]/float(turns_to_read)
            return output
        else:
            return self.phase_weight_function((1-len(self.register)),self.phase_shift_per_turn,delta_Phi)*self.register[0]


class CosineSumRegister(Register):
    def __init__(self,phase_shift_per_turn,delay, avg_length=1, position_phase_angle = 0, n_slices = None):
        super(self.__class__, self).__init__(self.phase_weight_function, self.delta_phi_calculator, phase_shift_per_turn,delay, avg_length, position_phase_angle, n_slices)

    def phase_weight_function(self,delay,phase_shift_per_turn,delta_phi):
        # delta_phi is betatron phase angle between reader and register
        return 2.*math.cos(delay*phase_shift_per_turn+delta_phi)

    def delta_phi_calculator(self,register_phase_angle,reader_phase_angle):
        delta_phi = register_phase_angle - reader_phase_angle


        #if delta_phi > 0:
        if delta_phi > pi/2.:
            #print "done"
            delta_phi = register_phase_angle - reader_phase_angle - 1. * self.phase_shift_per_turn

        return delta_phi