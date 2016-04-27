import itertools
import math
import copy
from collections import deque

import numpy as np
from scipy.constants import c, pi
import scipy.integrate as integrate
import scipy.special as special
import scipy.signal as signal


""" This file contains signal processors which can be used to process signals in the feedback module of PyHEADTAIL.

    A general requirement for the signal processor is that it is a class object which contains a function, namely,
    process(signal, slice_set). The input parameters of the function process(signal, slice_set) are a numpy array
    'signal' and a slice_set object from PyHEADTAIL. The function must return a numpy array with equal length to
    the input array.
"""

# TODO: add delay processor (ns scale)
# TODO: high pass filter
# TODO: Fix comments
# TODO: file read


class PickUp(object):
    """ A signal processor, which models realistic two plates pickup system, which has a finite noise level and
        bandwidth. The model assumes that signals from the plates vary to opposite polarities from the reference signal
        level. The signals of both plates pass separately ChargeWeighter, NoiseGenerator and LowpassFilter in order to
        simulate realistic levels of signal, noise and frequency response. The output signal is calculated from
        the ratio of a difference and sum of the signals. Signals below a given threshold level is set to zero
        in order to avoid high noise level at low input signal levels

        If the cut off frequency of the LowpassFilter is higher than 'sampling rate', a signal passes this model without
        changes. In other cases, a step response is faster than by using only a LowpassFilter but still finite.
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
    """ General class for linear signal processing. The signal is processed by calculating a dot product of a transfer
        matrix and a signal. The transfer matrix is produced from response function and (possible non uniform) z_bin_set
        by using generate_matrix function.
    """

    def __init__(self, norm_type=None, norm_range=None, bin_check = False, bin_middle = 'bin'):
        """
        :param scaling: Because integration by substitution doesn't work with np.quad (see quad_problem.ipynbl), it
            must be done by scaling integral limits. This parameter is a linear scaling coefficient of the integral
            limits. An ugly way which must be fixed.
        :param norm_type: Describes a normalization method for the transfer matrix
            'bunch_average': an average value over the bunch is equal to 1
            'fixed_average': an average value over a range given in a parameter norm_range is equal to 1
            'bunch_integral': an integral over the bunch is equal to 1
            'fixed_integral': an integral over a fixed range given in a parameter norm_range is equal to 1
            'matrix_sum': a sum over elements in the middle column of the matrix is equal to 1
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
    """ Return a signal, whose length corresponds to the input signal, but has been filled with an average value of
        the input signal. This is implemented by using an uniform matrix in LinearProcessor (response_function returns
        a constant value and sums of the rows in the matrix are normalized to be one).
    """
    def __init__(self,norm_type = 'matrix_sum', norm_range = None):
        super(self.__class__, self).__init__(norm_type, norm_range)

    def response_function(self, ref_bin_mid, ref_bin_from, ref_bin_to, bin_mid, bin_from, bin_to):
        return 1

class Bypass(LinearTransform):
    """ Return a signal, whose length corresponds to the input signal, but has been filled with an average value of
        the input signal. This is implemented by using an uniform matrix in LinearProcessor (response_function returns
        a constant value and sums of the rows in the matrix are normalized to be one).
    """
    def __init__(self,norm_type = 'matrix_sum', norm_range = None):
        super(self.__class__, self).__init__(norm_type, norm_range)

    def response_function(self, ref_bin_mid, ref_bin_from, ref_bin_to, bin_mid, bin_from, bin_to):
        if ref_bin_mid == bin_mid:
            return 1
        else:
            return 0

class Delay(LinearTransform):
    """ Delays signal. Delay in units of [second].
    """
    def __init__(self,delay, norm_type = None, norm_range = None):
        self.delay = delay
        super(self.__class__, self).__init__(norm_type, norm_range)

    def response_function(self, ref_bin_mid, ref_bin_from, ref_bin_to, bin_mid, bin_from, bin_to):

        if bin_from <= (ref_bin_from+self.delay*c) and bin_to >= (ref_bin_to+self.delay*c):
            print all
            return 1.
        elif bin_from >= (ref_bin_from + self.delay * c) and bin_to <= (ref_bin_to + self.delay * c):
            return (bin_to-bin_from) / float(ref_bin_to - ref_bin_from)

        elif bin_from <= (ref_bin_from + self.delay * c) < bin_to:
            return (bin_to - (ref_bin_from + self.delay * c)) / float(ref_bin_to - ref_bin_from)

        elif bin_from < (ref_bin_to + self.delay * c) <= bin_to:
            return ((ref_bin_to + self.delay * c) - bin_from) / float(ref_bin_to - ref_bin_from)
        else:
            return 0.


class PhaseLinearizedLowpass(LinearTransform):
    """ Phase linearized lowpass filter, which can be used to describe a frequency behavior of a kicker. A impulse response
        of a phase linearized lowpass filter is modified Bessel function of the second kind (np.special.k0).
        The transfer function has been derived by Gerd Kotzian.
    """

    # TODO: Add 2 pi?
    def __init__(self, f_cutoff, norm_type = 'matrix_sum', norm_range = None):
        self.scaling = f_cutoff/c
        self.norm_range_coeff = 10
        super(self.__class__, self).__init__(norm_type, norm_range)

    def response_function(self, ref_bin_mid, ref_bin_from, ref_bin_to, bin_mid, bin_from, bin_to):
        temp, _ = integrate.quad(self.transfer_function, self.scaling * (bin_from - ref_bin_mid),
                       self.scaling * (bin_to - ref_bin_mid))
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
        self.scaling = 2.*pi*f_cutoff/c
        super(self.__class__, self).__init__(norm_type, norm_range)

    def response_function(self, ref_bin_mid, ref_bin_from, ref_bin_to, bin_mid, bin_from, bin_to):
        temp, _ = integrate.quad(self.transfer_function, self.scaling * (bin_from - ref_bin_mid),
                       self.scaling * (bin_to - ref_bin_mid))
        return temp

    def transfer_function(self,x):
        if x < 0:
            return 0
        else:
            return math.exp(-1.*x)

class HighpassFilter(LinearTransform):
    """ Classical first order highpass filter (e.g. a RC filter)
    """
    def __init__(self, f_cutoff, norm_type = 'matrix_sum', norm_range = None):
        self.scaling = 2.*pi*f_cutoff/c
        super(self.__class__, self).__init__(norm_type, norm_range)

    def response_function(self, ref_bin_mid, ref_bin_from, ref_bin_to, bin_mid, bin_from, bin_to):
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
    """ A general class for signal weighing. A seed for the weight is a property of slices or the signal itself.
        The weight is calculated by passing the weight seed through weight_function.
    """

    # TODO: multiplier

    def __init__(self, seed, normalization = None, recalculate_multiplier = False):
        """
        :param weight_seed: 'bin_length', 'bin_midpoint', 'signal' or a property of a slice, which can be found
            from slice_set
        :param weight_normalization:
            'total_weight':  a sum of weight is equal to 1.
            'average_weight': an average weight over slices is equal to 1,
            'maximum_weight': a maximum weight value is equal to 1
            'minimum_weight': a minimum weight value is equal to 1
        :param: recalculate_weight: if True, the weight is recalculated every time when process() is called
        """

        self.seed = seed
        self.normalization = normalization
        self.recalculate_multiplier = recalculate_multiplier

        self.multiplier = None

    def multiplication_function(self, *args):
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
        else:
            norm_coeff = 1.

        self.multiplier = self.multiplier / norm_coeff


class ChargeWeighter(Multiplication):
    """ weight signal with charge (macroparticles) of slices
    """

    def __init__(self, normalization = 'maximum_weight'):
        super(self.__class__, self).__init__('n_macroparticles_per_slice', normalization)

    def multiplication_function(self,weight):
        return weight


class FermiDiracInverseWeighter(Multiplication):
    """ Use an inverse of the Fermi-Dirac distribution function to increase signal strength on edge of the bunch
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



class Addition(object):
    """ A general class for signal weighing. A seed for the weight is a property of slices or the signal itself.
        The weight is calculated by passing the weight seed through weight_function.
    """

    def __init__(self, seed, normalization = None, recalculate_addend = False):
        """
        :param weight_seed: 'bin_length', 'bin_midpoint', 'signal' or a property of a slice, which can be found
            from slice_set
        :param weight_normalization:
            'total_weight':  a sum of weight is equal to 1.
            'average_weight': an average weight over slices is equal to 1,
            'maximum_weight': a maximum weight value is equal to 1
            'minimum_weight': a minimum weight value is equal to 1
        :param: recalculate_weight: if True, the weight is recalculated every time when process() is called
        """

        self.seed = seed
        self.normalization = normalization
        self.recalculate_addend = recalculate_addend

        self.addend = None

    def addend_function(self, *args):
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
    """ Add noise to a signal. The noise level is given as RMS value of an absolute level (reference_level = 'absolute'),
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
    """ A general class for a signal register. A signal is stored to the register, when the function process() is
        called. The register is iterable and returns values which have been kept in register longer than
        delay requires. Normally this means that a number of returned signals corresponds to a paremeter avg_length, but
        it is less during the first turns. Phase shifts caused by delays and positions can be taken into account
        by using information stored to variables phase_shift_per_turn, position and reader_position. The variable
        reader_position is updated, when it is given in a call of the iterator, i.e. by using
                for value in register(reader_position):
            instead of
                for value in register:

        When the register is a part of a signal processor chain, the function process() must return np.array() which
        length corresponds to the length of the input signal. The class can and must be customized by overwriting
        the iterator function next().

    """

    def __init__(self,delay, avg_length, phase_shift_per_turn, position, n_slices, in_processor_chain, combine_prev = False):
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
        self.combine_prev = combine_prev


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
            # print (np.zeros(len(self.register[0])),None,self.position)
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

    def combine(self,x1,x2,reader_position,x_to_xp = False):

        return x1


class VectorSumRegister(Register):
    # A plain register, which does not modify signals. The function process() returns the latest value in the register.
    # In the other cases, next() returns unmodified values.

    def __init__(self,delay, avg_length, phase_shift_per_turn, position=None, n_slices=None, in_processor_chain=True):
        self.type = 'plain'
        super(self.__class__, self).__init__(delay, avg_length, phase_shift_per_turn, position, n_slices, in_processor_chain)

    def combine(self,x1,x2,reader_position,x_to_xp = False):
        if (len(x1) < 4) or (len(x2) < 4):
            print "len(x1)=" + str(len(x1)) + " and len(x2)=" + str(len(x2))

            print x1
            print x2
        # calculates a vector in position x1
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
    # A register which uses Hilber transform to calculate a betatron oscillation amplitude. The register returns in the
    # both cases (call of process() and iteration) a single value, which is an averaged oscillation amplitudevalue.
    # Note that avg_length must be sufficient (e.g. > X) in order to do a reliable calculation

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

        re = x1[0]
        im = x1[1]
        # turns the vector to the reader's position
        # turns the vector to the reader's position
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
