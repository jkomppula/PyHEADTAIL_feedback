import itertools
import math
import copy
from collections import deque
from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.constants import c, pi
import scipy.integrate as integrate
import scipy.special as special
import scipy.signal as signal


""" This file contains signal processors which can be used in the feedback module in PyHEADTAIL.

    A general requirement for the signal processor is that it is a class object containing a function, namely,
    process(signal, slice_set). The input parameters for the function process(signal, slice_set) are a numpy array
    'signal' and a slice_set object of PyHEADTAIL. The function must return a numpy array with equal length to
    the input array.

    The signals processors in this file are based on four abstract classes;
        1) in LinearTransform objects the input signal is multiplied with a matrix.
        2) in Multiplication objects the input signal is multiplied with an array with equal length to the input array
        3) in Addition objects to the input signal is added an array with equal length to the input array
        4) A normal signal processor doesn't store a signal (in terms of process() calls). Processors buffering,
           registering and/or delaying signals are namely Registers. The Registers have following properties in addition
           to the normal processor:
            a) the object is iterable
            b) the object contains a function namely combine(*args), which combines two signals returned by iteration
               together

"""
# TODO: file read


class PickUp(object):
    """ A signal processor, which models a realistic two plates pickup system, which has a finite noise level and
        bandwidth. The model assumes that signals from the plates vary to opposite polarities from the reference signal
        level. The signals of both plates pass separately ChargeWeighter, NoiseGenerator and LowpassFilter in order to
        simulate realistic levels of signal, noise and frequency response. The output signal is calculated from
        the ratio of a difference and sum signals of the plates. Signals below a given threshold level is set to zero
        in order to avoid high noise level at low input signal levels

        If the cut off frequency of the LowpassFilter is higher than 'the sampling rate', a signal passes this model
        without changes. In other cases, a step response is faster than by using only a LowpassFilter but still finite.
    """

    def __init__(self,RMS_noise_level,f_cutoff, threshold_level):

        self.threshold_level = threshold_level

        self.noise_generator = NoiseGenerator(RMS_noise_level)
        self.filter = LowpassFilter(f_cutoff)
        self.charge_weighter = ChargeWeighter()

    def process(self,signal,slice_set):

        reference_level = 1

        signal_A = (reference_level + np.array(signal))
        signal_A = self.charge_weighter.process(signal_A,slice_set)
        signal_A = self.noise_generator.process(signal_A,slice_set)
        signal_A = self.filter.process(signal_A,slice_set)

        signal_B = (reference_level - np.array(signal))
        signal_B = self.charge_weighter.process(signal_B,slice_set)
        signal_B = self.noise_generator.process(signal_B,slice_set)
        signal_B = self.filter.process(signal_B,slice_set)

        # sets signals below the threshold level to 0. Multiplier 2 to the threshold level comes the fact that
        # the signal difference is two times the original level and the threshold level refers to the original signal
        signal_diff = signal_A - signal_B
        signal_diff[np.absolute(signal_diff) < 2.*self.threshold_level] = 0.

        # in order to avoid 0/0, also sum signals below the threshold level have been set to 1
        signal_sum = (signal_A + signal_B)
        signal_sum[np.absolute(signal_diff) < 2.*self.threshold_level] = 1.

        return reference_level * signal_diff / signal_sum

class LinearTransform(object):
    __metaclass__ = ABCMeta
    """ An abstract class for signal processors which are based on linear transformation. The signal is processed by
        calculating a dot product of a transfer matrix and a signal. The transfer matrix is produced with an abstract
        method, namely response_function(*args), which returns an elements of the matrix (a ref_bin affection to a bin)
    """

    def __init__(self, norm_type=None, norm_range=None, bin_check = False, bin_middle = 'bin'):
        """

        :param norm_type: Describes a normalization method for the transfer matrix
            'bunch_average': an average value over the bunch is equal to 1
            'fixed_average': an average value over a range given in a parameter norm_range is equal to 1
            'bunch_integral': an integral over the bunch is equal to 1
            'fixed_integral': an integral over a fixed range given in a parameter norm_range is equal to 1
            'matrix_sum': a sum over elements in the middle column of the matrix is equal to 1
            None: no normalization
        :param norm_range: Normalization length in cases of self.norm_type == 'fixed_length_average' or
            self.norm_type == 'fixed_length_integral'
        :param bin_check: if True, a change of the bin_set is checked every time process() is called and matrix is
            recalculated if any change is found
        :param bin_middle: defines if middle points of the bins are determined by a middle point of the bin
            (bin_middle = 'bin') or an average place of macro particles (bin_middle = 'particles')
        """

        self.norm_type = norm_type
        self.norm_range = norm_range
        self.bin_check = bin_check
        self.bin_middle = bin_middle

        self.z_bin_set = None
        self.matrix = None

        self.recalculate_matrix = True

    @abstractmethod
    def response_function(self, ref_bin_mid, ref_bin_from, ref_bin_to, bin_mid, bin_from, bin_to):
        # Impulse response function of the processor
        pass

    def process(self,signal,slice_set, *args):

        # check if the bin set is changed
        if self.bin_check:
            self.recalculate_matrix = self.check_bin_set(slice_set.z_bins)

        # recalculte the matrix if necessary
        if self.recalculate_matrix:
            self.recalculate_matrix = False
            if self.bin_middle == 'particles':
                bin_midpoints = np.array(copy.copy(slice_set.mean_z))
            elif self.bin_middle == 'bin':
                bin_midpoints = np.array([(i+j)/2. for i, j in zip(slice_set.z_bins, slice_set.z_bins[1:])])

            self.generate_matrix(slice_set.z_bins,bin_midpoints)

        # process the signal
        return np.dot(self.matrix,signal)

    def check_bin_set(self,z_bin_set):

        if self.z_bin_set is None:
            self.z_bin_set = copy.copy(z_bin_set)
            return True

        else:
            changed = False
            for old, new in itertools.izip(self.z_bin_set,z_bin_set):
                if old != new:
                    changed = True
                    self.z_bin_set = copy.copy(z_bin_set)
                    break
            return changed

    def print_matrix(self):
        for row in self.matrix:
            print "[",
            for element in row:
                print "{:6.3f}".format(element),
            print "]"

    def generate_matrix(self,bin_set, bin_midpoints):

        if self.norm_type == 'bunch_average':
            self.norm_coeff = bin_set[-1] - bin_set[0]
        elif self.norm_type == 'fixed_average':
            self.norm_coeff = self.norm_range[1] - self.norm_range[0]
        elif self.norm_type == 'bunch_integral':
            center_idx = math.floor(len(bin_midpoints) / 2)
            self.norm_coeff = self.response_function(bin_midpoints[center_idx], bin_set[center_idx],
                                                    bin_set[center_idx + 1], bin_midpoints[center_idx],
                                                    bin_set[0], bin_set[-1])
        elif self.norm_type == 'fixed_integral':
            center_idx = math.floor(len(bin_midpoints) / 2)
            self.norm_coeff = self.response_function(bin_midpoints[center_idx], bin_set[center_idx],
                                                    bin_set[center_idx + 1], bin_midpoints[center_idx],
                                                    self.norm_range[0], self.norm_range[-1])
        elif self.norm_type == 'matrix_sum':
            self.norm_coeff = 0
            center_idx = math.floor(len(bin_midpoints) / 2)
            for i, midpoint in enumerate(bin_midpoints):
                self.norm_coeff += self.response_function(bin_midpoints[center_idx], bin_set[center_idx],
                                                    bin_set[center_idx + 1], midpoint, bin_set[i], bin_set[i + 1])
        elif self.norm_type is None:
            self.norm_coeff = 1

        self.matrix = np.identity(len(bin_midpoints))

        for i, midpoint_i in enumerate(bin_midpoints):
            for j, midpoint_j in enumerate(bin_midpoints):
                    self.matrix[j][i] = self.response_function(midpoint_i,bin_set[i],bin_set[i+1],midpoint_j,bin_set[j]
                                                               ,bin_set[j+1]) / float(self.norm_coeff)

class Averager(LinearTransform):
    """ Returns a signal, which consists an average value of the input signal. A sums of the rows in the matrix
    are normalized to be one (i.e. a sum of the input signal doesn't change).
    """

    def __init__(self,norm_type = 'matrix_sum', norm_range = None):
        super(self.__class__, self).__init__(norm_type, norm_range)

    def response_function(self, ref_bin_mid, ref_bin_from, ref_bin_to, bin_mid, bin_from, bin_to):
        return 1

class Bypass(LinearTransform):
    """ Passes a signal without change (an identity matrix).
    """
    def __init__(self,norm_type = None, norm_range = None):
        super(self.__class__, self).__init__(norm_type, norm_range)

    def response_function(self, ref_bin_mid, ref_bin_from, ref_bin_to, bin_mid, bin_from, bin_to):
        if ref_bin_mid == bin_mid:
            return 1
        else:
            return 0

class Delay(LinearTransform):
    """ Delays signal in units of [second].
    """
    def __init__(self,delay, norm_type = None, norm_range = None):
        self.delay = delay
        super(self.__class__, self).__init__(norm_type, norm_range)

    def response_function(self, ref_bin_mid, ref_bin_from, ref_bin_to, bin_mid, bin_from, bin_to):
        
        return self.CDF(bin_to-self.delay*c, ref_bin_from, ref_bin_to)-self.CDF(bin_from-self.delay*c, ref_bin_from, ref_bin_to)

    def CDF(self,x,ref_bin_from, ref_bin_to):
        if x <= ref_bin_from:
            return 0.
        elif x < ref_bin_to:
            return (x-ref_bin_from)/float(ref_bin_to-ref_bin_from)
        else:
            return 1.

class PhaseLinearizedLowpass(LinearTransform):
    """ Phase linearized lowpass filter, which can be used to describe a frequency behavior of a kicker. A impulse response
        of a phase linearized lowpass filter is modified Bessel function of the second kind (np.special.k0).
        The transfer function has been derived by Gerd Kotzian.
    """

    def __init__(self, f_cutoff, norm_type = 'matrix_sum', norm_range = None):
        self.f_cutoff = f_cutoff
        self.norm_range_coeff = 10
        super(self.__class__, self).__init__(norm_type, norm_range)

    def response_function(self, ref_bin_mid, ref_bin_from, ref_bin_to, bin_mid, bin_from, bin_to):
        # Frequency scaling must be done by scaling integral limits, because integration by substitution doesn't work
        # with np.quad (see quad_problem.ipynbl). An ugly way could be fixed.
        # TODO: Add 2 pi?
        scaling = self.f_cutoff / c
        temp, _ = integrate.quad(self.transfer_function, scaling * (bin_from - ref_bin_mid),
                       scaling * (bin_to - ref_bin_mid))
        return temp

    def transfer_function(self,x):
        if x == 0:
            return 0
        else:
            return special.k0(abs(x))


class LowpassFilter(LinearTransform):
    """ Classical first order lowpass filter (e.g. a RC filter), whose impulse response can be described as exponential
        decay.
    """
    def __init__(self, f_cutoff, norm_type = 'matrix_sum', norm_range = None):
        self.f_cutoff = f_cutoff
        super(self.__class__, self).__init__(norm_type, norm_range)

    def response_function(self, ref_bin_mid, ref_bin_from, ref_bin_to, bin_mid, bin_from, bin_to):
        # Frequency scaling must be done by scaling integral limits, because integration by substitution doesn't work
        # with np.quad (see quad_problem.ipynbl). An ugly way could be fixed.
        scaling = 2.*pi*self.f_cutoff/c
        temp, _ = integrate.quad(self.transfer_function, scaling * (bin_from - ref_bin_mid),
                       scaling * (bin_to - ref_bin_mid))
        return temp

    def transfer_function(self,x):
        if x < 0:
            return 0
        else:
            return math.exp(-1.*x)

class HighpassFilter(LinearTransform):
    """ Classical first order highpass filter (e.g. a RC filter)
    """
    def __init__(self, f_cutoff, norm_type = None, norm_range = None):
        self.f_cutoff = f_cutoff
        super(self.__class__, self).__init__(norm_type, norm_range)

    def response_function(self, ref_bin_mid, ref_bin_from, ref_bin_to, bin_mid, bin_from, bin_to):
        # Frequency scaling must be done by scaling integral limits, because integration by substitution doesn't work
        # with np.quad (see quad_problem.ipynbl). An ugly way could be fixed.
        scaling = 2.*pi*self.f_cutoff/c

        temp, _ = integrate.quad(self.transfer_function, self.scaling * (bin_from - ref_bin_mid),
                       self.scaling * (bin_to - ref_bin_mid))

        if ref_bin_mid == bin_mid:
            temp += 1.

        return temp

    def transfer_function(self,x):
        if x < 0:
            return 0
        else:
            return -1.*math.exp(-1.*x)

class Bypass_Fast(object):
    def process(self,signal, *args):
        return signal


class Multiplication(object):
    __metaclass__ = ABCMeta
    """ An abstract class which multiplies the input signal by an array. The multiplier array is produced by taking
        a slice property (determined in the input parameter 'seed') and passing it through the abstract method, namely
        multiplication_function(seed).
    """
    def __init__(self, seed, normalization = None, recalculate_multiplier = False):
        """
        :param seed: 'bin_length', 'bin_midpoint', 'signal' or a property of a slice, which can be found
            from slice_set
        :param normalization:
            'total_weight':  a sum of the multiplier array is equal to 1.
            'average_weight': an average in  the multiplier array is equal to 1,
            'maximum_weight': a maximum value in the multiplier array value is equal to 1
            'minimum_weight': a minimum value in the multiplier array value is equal to 1
        :param: recalculate_weight: if True, the weight is recalculated every time when process() is called
        """

        self.seed = seed
        self.normalization = normalization
        self.recalculate_multiplier = recalculate_multiplier

        self.multiplier = None

    @abstractmethod
    def multiplication_function(self, seed):
        pass

    def process(self,signal,slice_set):

        if (self.multiplier is None) or self.recalculate_multiplier:
            self.calculate_multiplier(signal,slice_set)

        return self.multiplier*signal

    def calculate_multiplier(self,signal,slice_set):
        if self.seed == 'bin_length':
            bin_set = slice_set.z_bins
            self.multiplier = np.array([(j-i) for i, j in zip(bin_set, bin_set[1:])])
        elif self.seed == 'bin_midpoint':
            bin_set = slice_set.z_bins
            self.multiplier = np.array([(i+j)/2 for i, j in zip(bin_set, bin_set[1:])])
        elif self.seed == 'signal':
            self.multiplier = np.array(signal)
        else:
            self.multiplier = np.array(getattr(slice_set,self.seed))

        self.multiplier = self.multiplication_function(self.multiplier)

        if self.normalization == 'total_weight':
            norm_coeff = float(np.sum(self.multiplier))
        elif self.normalization == 'average_weight':
            norm_coeff = float(np.sum(self.multiplier))/float(len(self.multiplier))
        elif self.normalization == 'maximum_weight':
            norm_coeff = float(np.max(self.multiplier))
        elif self.normalization == 'minimum_weight':
            norm_coeff = float(np.min(self.multiplier))
        elif self.normalization == None:
            norm_coeff = 1.

        self.multiplier = self.multiplier / norm_coeff


class ChargeWeighter(Multiplication):
    """ weights signal with charge (macroparticles) of slices
    """

    def __init__(self, normalization = 'maximum_weight'):
        super(self.__class__, self).__init__('n_macroparticles_per_slice', normalization)

    def multiplication_function(self,weight):
        return weight


class FermiDiracInverseWeighter(Multiplication):
    """ Use an inverse of the Fermi-Dirac distribution function to increase signal strength on the edges of the bunch
    """

    def __init__(self,bunch_length,bunch_decay_length,maximum_weight = 10):
        """
        :param bunch_length: estimated width of the bunch
        :param bunch_decay_length: slope of the function on the edge of the bunch. Smaller value, steeper slope.
        :param maximum_weight: maximum value of the weight
        """
        self.bunch_length = bunch_length
        self.bunch_decay_length = bunch_decay_length
        self.maximum_weight=maximum_weight
        super(self.__class__, self).__init__('bin_midpoint', 'minimum_weight')

    def multiplication_function(self,weight):
        weight = np.exp((np.absolute(weight)-self.bunch_length/2.)/float(self.bunch_decay_length))+ 1.
        weight = np.clip(weight,1.,self.maximum_weight)
        return weight


class NoiseGate(Multiplication):
    """ Passes a signal which level is greater/less than the threshold level.
    """

    def __init__(self,threshold, operator = 'greater', threshold_ref = 'amplitude'):

        self.threshold = threshold
        self.operator = operator
        self.threshold_ref = threshold_ref
        super(self.__class__, self).__init__('signal', None,recalculate_multiplier = True)

    def multiplication_function(self, seed):
        multiplier = np.zeros(len(seed))

        if self.threshold_ref == 'amplitude':
            comparable = np.abs(seed)
        elif self.threshold_ref == 'absolute':
            comparable = seed

        if self.operator == 'greater':
            multiplier[comparable > self.threshold] = 1
        elif self.operator == 'less':
            multiplier[comparable < self.threshold] = 1

        return multiplier


class Addition(object):
    __metaclass__ = ABCMeta
    """ An abstract class which adds an array to the input signal. The addend array is produced by taking
        a slice property (determined in the input parameter 'seed') and passing it through the abstract method, namely
        addend_function(seed).
    """

    def __init__(self, seed, normalization = None, recalculate_addend = False):
        """
        :param seed: 'bin_length', 'bin_midpoint', 'signal' or a property of a slice, which can be found
            from slice_set
        :param normalization:
            'total_weight':  a sum of the multiplier array is equal to 1.
            'average_weight': an average in  the multiplier array is equal to 1,
            'maximum_weight': a maximum value in the multiplier array value is equal to 1
            'minimum_weight': a minimum value in the multiplier array value is equal to 1
        :param: recalculate_weight: if True, the weight is recalculated every time when process() is called
        """

        self.seed = seed
        self.normalization = normalization
        self.recalculate_addend = recalculate_addend

        self.addend = None

    @abstractmethod
    def addend_function(self, seed):
        pass

    def process(self,signal,slice_set):

        if (self.addend is None) or self.recalculate_addend:
            self.calculate_addend(signal,slice_set)

        return signal + self.addend

    def calculate_addend(self,signal,slice_set):
        if self.seed == 'bin_length':
            bin_set = slice_set.z_bins
            self.addend = np.array([(j-i) for i, j in zip(bin_set, bin_set[1:])])
        elif self.seed == 'bin_midpoint':
            bin_set = slice_set.z_bins
            self.addend = np.array([(i+j)/2 for i, j in zip(bin_set, bin_set[1:])])
        elif self.seed == 'signal':
            self.addend = np.array(signal)
        else:
            self.addend = np.array(getattr(slice_set,self.seed))

        self.addend = self.addend_function(self.addend)

        if self.normalization == 'total':
            norm_coeff = float(np.sum(self.addend))
        elif self.normalization == 'average':
            norm_coeff = float(np.sum(self.addend))/float(len(self.addend))
        elif self.normalization == 'maximum':
            norm_coeff = float(np.max(self.addend))
        elif self.normalization == 'minimum':
            norm_coeff = float(np.min(self.addend))
        else:
            norm_coeff = 1.

        self.addend = self.addend / norm_coeff

class NoiseGenerator(Addition):
    """ Adds noise to a signal. The noise level is given as RMS value of the absolute level (reference_level = 'absolute'),
        a relative RMS level to the maximum signal (reference_level = 'maximum') or a relative RMS level to local
        signal values (reference_level = 'local'). Options for the noise distribution are a Gaussian normal distribution
        (distribution = 'normal') and an uniform distribution (distribution = 'uniform')
    """

    def __init__(self,RMS_noise_level,reference_level = 'absolute', distribution = 'normal'):

        self.RMS_noise_level = RMS_noise_level
        self.reference_level = reference_level
        self.distribution = distribution

        super(self.__class__, self).__init__('signal', None, True)

    def addend_function(self,seed):

        randoms = np.zeros(len(seed))

        if self.distribution == 'normal' or self.distribution is None:
            randoms = np.random.randn(len(seed))
        elif self.distribution == 'uniform':
            randoms = 1./0.577263*(-1.+2.*np.random.rand(len(seed)))

        if self.reference_level == 'absolute':
            addend = self.RMS_noise_level*randoms
        elif self.reference_level == 'maximum':
            addend = self.RMS_noise_level*np.max(seed)*randoms
        elif self.reference_level == 'local':
            addend = signal*self.RMS_noise_level*randoms

        return addend

class Register(object):
    __metaclass__ = ABCMeta

    """ An abstract class for a signal register. A signal is stored to the register, when the function process() is
        called. The register is iterable and returns values which have been kept in register longer than
        delay requires. Normally this means that a number of returned signals corresponds to a paremeter avg_length, but
        it is less during the first turns. The values from the register can be calculated together by using a abstract
        function combine(*). It manipulates values (in terms of a phase advance) such way they can be calculated
        together in the reader position.

        When the register is a part of a signal processor chain, the function process() returns np.array() which
        is an average of register values determined by a paremeter avg_length. The exact functionality of the register
        is determined by in the abstract iterator combine(*args).

    """

    def __init__(self,delay, avg_length, phase_shift_per_turn, position, n_slices, in_processor_chain):
        """
        :param delay: a delay between storing to reading values  in turns
        :param avg_length: a number of register values are averaged
        :param phase_shift_per_turn: a betatron phase shift per turn
        :param position: a betatron position (angle) of the register from a reference point
        :param n_slices: a length of a signal, which is returned if the register is empty
        :param in_processor_chain: if True, process() returns a signal
        """
        self.delay = delay
        self.avg_length = avg_length
        self.phase_shift_per_turn = phase_shift_per_turn
        self.position = position
        self.in_processor_chain = in_processor_chain


        self.max_reg_length = self.delay+self.avg_length
        self.register = deque()

        self.n_iter_left = -1

        self.reader_position = None

        if n_slices is not None:
            self.register.append(np.zeros(n_slices))

    def __iter__(self):
        # calculates a maximum number of iterations. If there is no enough values in the register, sets -1, which
        # indicates that next() might return zero value

        self.n_iter_left =  len(self)
        if self.n_iter_left == 0:
            self.n_iter_left = -1
        return self

    def __len__(self):
        # returns a number of signals in the register after delay
        return max((len(self.register) - self.delay), 0)

    def next(self):
        if self.n_iter_left == 0:
            raise StopIteration
        elif self.n_iter_left == -1:
            self.n_iter_left = 0
            return (np.zeros(len(self.register[0])),None,0,self.position)
        else:
            delay = -1. * (len(self.register) - self.n_iter_left) * self.phase_shift_per_turn
            self.n_iter_left -= 1
            return (self.register[self.n_iter_left],None,delay,self.position)

    def process(self,signal, *args):

        self.register.append(signal)

        if len(self.register) > self.max_reg_length:
            self.register.popleft()

        if self.in_processor_chain == True:
            temp_signal = np.zeros(len(signal))
            if len(self) > 0:

                prev = (np.zeros(len(self.register[0])),None,0,self.position)

                for value in self:
                    combined = self.combine(value,prev,None)
                    prev = value
                    temp_signal += combined[0] / float(len(self))

            return temp_signal

    @abstractmethod
    def combine(self,x1,x2,reader_position,x_to_xp = False):

        pass


class VectorSumRegister(Register):

    def __init__(self,delay, avg_length, phase_shift_per_turn, position=None, n_slices=None, in_processor_chain=True):
        self.type = 'plain'
        super(self.__class__, self).__init__(delay, avg_length, phase_shift_per_turn, position, n_slices, in_processor_chain)

    def combine(self,x1,x2,reader_position,x_to_xp = False):
        # determines a complex number representation from two signals (e.g. from two pickups or different turns), by using
        # knowledge about phase advance between signals. After this turns the vector to the reader's phase
        # TODO: Why not x2[3]-x1[3]?
        phi_x1_x2 = x1[3]-x2[3]
        # print 'x1: ' + str(x1[3]) + ' x2: ' + str(x2[3]) + ' diff:' + str(phi_x1_x2)
        s = np.sin(phi_x1_x2)
        c = np.cos(phi_x1_x2)
        re = x1[0]
        im = (c*x1[0]-x2[0])/float(s)

        # turns the vector to the reader's position
        delta_phi = x1[2]
        if reader_position is not None:
            delta_position = x1[3] - reader_position
            delta_phi += delta_position
            if delta_position > 0:
                delta_phi -= self.phase_shift_per_turn
            if x_to_xp == True:
                delta_phi -= pi/2.

        s = np.sin(delta_phi)
        c = np.cos(delta_phi)

        return np.array([c*re-s*im,s*re+c*im])


class CosineSumRegister(Register):
    """ Returns register values by multiplying the values with a cosine of the betatron phase angle from the reader.
        If there are multiple values in different phases, the sum approaches a value equal to half of the displacement
        in the reader's position.

        The function process() returns a value, which is an average of the register values (after delay determined by
        the parameter avg_length)
    """
    def __init__(self,delay, avg_length, phase_shift_per_turn, position=None, n_slices=None, in_processor_chain=True):
        self.type = 'cosine'
        super(self.__class__, self).__init__(delay, avg_length, phase_shift_per_turn, position, n_slices, in_processor_chain)

    def combine(self,x1,x2,reader_position,x_to_xp = False):
        #print "x_to_xp: " + str(x_to_xp)
        delta_phi = x1[2]
        if reader_position is not None:
            delta_position = self.position - reader_position
            delta_phi += delta_position
            if delta_position > 0:
                delta_phi -= self.phase_shift_per_turn
            if x_to_xp == True:
                delta_phi -= pi/2.

        return np.array([2.*math.cos(delta_phi)*x1[0],None])


class HilbertRegister(Register):
    # uses Hilbert transform to calculate a complex number representation for each value in the register. After this
    # turns all vectors to same direction and returns an average of these vectors. Note that avg_length must be
    # sufficient (e.g. >= 7) in order to do a reliable calculations.

    # DEV NOTES: phase rotation is messy thing. I don't understand exactly (yet) why this is working, but probably
    # because of beam phase rotation and Hilbert transform use different coordinate systems. Thus, there is a minus sign
    # in the front of the imaginary part of the outgoing vector and the vector has been rotated to different directions,
    # when average of vectors is calculated in next() and combine() functions.

    # FIXME:

    def __init__(self,delay, avg_length, phase_shift_per_turn, position=None, n_slices=None, in_processor_chain=True):
        super(self.__class__, self).__init__(delay, avg_length, phase_shift_per_turn, position, n_slices, in_processor_chain)

    def next(self):

        signal_length = len(self.register[0])
        if self.n_iter_left == 0:
            raise StopIteration
        elif len(self.register) < self.max_reg_length:
            self.n_iter_left = 0
            return (np.zeros(signal_length),np.zeros(signal_length),0,self.position)
        else:
            self.n_iter_left = 0

            if self.delay == 0:
                temp_data = np.array(self.register)
            else:
                temp_data = np.array(self.register)[:-1*self.delay]

            re = []
            im = []

            for i in range(signal_length):
                h = signal.hilbert(temp_data[:, i])
                re.append(np.real(h))
                im.append(np.imag(h))

            re = np.array(re)
            im = np.array(im)

            total_re = np.zeros(len(re))
            total_im = np.zeros(len(re))

            counter = 0

            edge_skip = 0

            for i in range(edge_skip,(len(re[0])-edge_skip)):

                delay = 1. * float(self.avg_length-i-1) * self.phase_shift_per_turn
                s = np.sin(delay)
                c = np.cos(delay)
                re_t = re[:,i]
                im_t = im[:,i]
                total_re += c * re_t - s * im_t
                total_im += s * re_t + s * im_t
                counter +=1

            total_re /= float(counter)
            total_im /= float(counter)

            delay = -1. * float(self.delay) * self.phase_shift_per_turn
            return (total_re, -1.*total_im, delay, self.position)

    def combine(self,x1,x2,reader_position,x_to_xp = False):
        # turns vector x1 to the readers phase

        re = x1[0]
        im = x1[1]
        delta_phi = x1[2]
        if reader_position is not None:
            delta_position = x1[3] - reader_position
            delta_phi += delta_position
            if delta_position > 0:
                delta_phi -= self.phase_shift_per_turn
            if x_to_xp == True:
                delta_phi -= pi / 2.

        s = np.sin(delta_phi)
        c = np.cos(delta_phi)

        return np.array([c * re - s * im, s * re + c * im])
