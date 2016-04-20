import itertools
import math
import copy
from collections import deque

import numpy as np
from scipy.constants import c, pi
import scipy.integrate as integrate
import scipy.special as special
import scipy.signal as signal


""" This file contains signal processors whom can be used to process signal in the feedback module of PyHEADTAIL.

    A general requirement for the signal processor is that it is a class object which contains a function, namely,
    process(signal, slice_set). The input parameters of the function process(signal, slice_set) are a numpy array
    'signal' and a slice_set object from PyHEADTAIL. The function must return a numpy array with equal length to
    the input array.
"""

# TODO: add delay processor (ns scale)
# TODO: similar register as used in SPS
# TODO: alternative implementation by using convolution or FFT?0,


class PickUp(object):
    """ Model for a realistic two plates pickup system, which has a finite noise level and bandwidth.
        The model assumes that signals from the plates vary to opposite polarities from the reference signal
        level. The reference signal level has been chosen to be 1 in order to avoid normalization at the end.
        The signals of both plates pass separately ChargeWeighter, NoiseGenerator and LowpassFilter in order to
        simulate realistic levels of signal, noise and frequency response. The output signal is calculated by
        using the ratio of a difference and sum of the signals.

        If the cut off frequency of the LowpassFilter is higher than 'sampling rate', a signal passes this without
        changes. In other cases, a step response is faster than by using only a LowpassFilter but finite.

        In principle, there are no bugs in the algorithm, but it is NOT RECOMMENDED TO USE it at the moment.
        Any noise outside the bunch is significantly amplified and also the trailing edge of the signal does not
        decay to zero due to infinite long exponential decay of the lowpass filters. These problems will be solved.
    """
        # TODO: Find a solution to an infinite long decay of signals
        # TODO: Ask about noise reduction outside of the bunch

    def __init__(self,RMS_noise_level,f_cutoff):

        self.noise_generator = NoiseGenerator(RMS_noise_level)
        self.filter = LowpassFilter(f_cutoff)
        self.charge_weighter = ChargeWeighter()

    def process(self,signal,slice_set):

        reference_level = 1.

        signal_1 = (reference_level + np.array(signal))
        signal_2 = (reference_level - np.array(signal))
        signal_1 = self.charge_weighter.process(signal_1,slice_set)
        signal_2 = self.charge_weighter.process(signal_2,slice_set)

        signal_1 = self.noise_generator.process(signal_1,slice_set)
        signal_1 = self.filter.process(signal_1,slice_set)

        signal_2 = self.noise_generator.process(signal_2,slice_set)
        signal_2 = self.filter.process(signal_2,slice_set)

#        print signal
#        print (reference_level*(signal_1-signal_2)/(signal_1+signal_2))/signal

        return reference_level*(signal_1-signal_2)/(signal_1+signal_2)


class NoiseGenerator(object):
    """ Add noise to a signal. The noise level is given as RMS value of an absolute level (reference_level = 'absolute'),
        a relative RMS level to the maximum signal (reference_level = 'maximum') or a relative RMS level to local
        signal values (reference_level = 'local'). Options for the noise distribution are a Gaussian normal distribution
        (distribution = 'normal') and an uniform distribution (distribution = 'uniform')
    """

    def __init__(self,RMS_noise_level,reference_level = 'absolute', distribution = 'normal'):

        self.RMS_noise_level = RMS_noise_level
        self.reference_level = reference_level
        self.distribution = distribution

        super(self.__class__, self).__init__()

    def process(self,signal,slice_set):

        randoms = np.zeros(len(signal))

        if self.distribution == 'normal' or self.distribution is None:
            randoms = np.random.randn(len(signal))
        elif self.distribution == 'uniform':
            randoms = 1./0.577263*(-1.+2.*np.random.rand(len(signal)))

        if self.reference_level == 'absolute':
            signal += self.RMS_noise_level*randoms
        elif self.reference_level == 'maximum':
            signal += self.RMS_noise_level*np.max(signal)*randoms
        elif self.reference_level == 'local':
            signal += signal*self.RMS_noise_level*randoms

        return signal


class LinearProcessor(object):
    """ General class for linear signal processing. The signal is processed by calculating a dot product of a transfer
        matrix and a signal. The transfer matrix is produced from response function and (possible non uniform) z_bin_set
        by using generate_matrix function.
    """

    def __init__(self, scaling=1., norm_type=None, norm_range=None, bin_check = False, bin_middle = 'bin'):
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

        self.scaling = scaling
        self.norm_type = norm_type
        self.norm_range = norm_range
        self.bin_check = bin_check
        self.bin_middle = bin_middle

        self.z_bin_set = None
        self.matrix = None

        self.recalculate_matrix = True

    def response_function(self, *args):
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
            else:
                bin_midpoints = np.array([(i+j)/2. for i, j in zip(slice_set.z_bins, slice_set.z_bins[1:])])

            self.matrix = self.generate_matrix(slice_set.z_bins,bin_midpoints)

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
        self.norm_coeff = 1

        if self.norm_type == 'bunch_average':
            self.norm_coeff=bin_set[-1]-bin_set[0]
        elif self.norm_type == 'fixed_average':
            self.norm_coeff=self.norm_range[1]-self.norm_range[0]
        elif self.norm_type == 'bunch_integral':
            self.norm_coeff, _ = integrate.quad(self.response_function, self.scaling*bin_set[0], self.scaling*bin_set[-1])
        elif self.norm_type == 'fixed_integral':
            self.norm_coeff, _ = integrate.quad(self.response_function, self.scaling*self.norm_range[0], self.scaling*self.norm_range[1])
        elif self.norm_type == 'matrix_sum':
            center_idx = math.floor(len(bin_midpoints)/2)
            self.norm_coeff, _ = integrate.quad(self.response_function,self.scaling*(bin_set[0]-bin_midpoints[center_idx]),self.scaling*(bin_set[-1]-bin_midpoints[center_idx]))

        matrix = np.identity(len(bin_midpoints))

        for i, midpoint in enumerate(bin_midpoints):
                for j in range(len(bin_midpoints)):

                    temp, _ = integrate.quad(self.response_function,self.scaling*(bin_set[j]-midpoint),self.scaling*(bin_set[j+1]-midpoint))
                    matrix[j][i] = temp/self.norm_coeff

        return matrix


class Averager(LinearProcessor):
    """ Return a signal, whose length corresponds to the input signal, but has been filled with an average value of
        the input signal. This is implemented by using an uniform matrix in LinearProcessor (response_function returns
        a constant value and sums of the rows in the matrix are normalized to be one).
    """
    def __init__(self,norm_type = 'matrix_sum', norm_range = None):
        super(self.__class__, self).__init__(1, norm_type, norm_range)

    def response_function(self,x):
        return 1


class PhaseLinearizedLowpass(LinearProcessor):
    """ Phase linearized lowpass filter, which can be used to describe a frequency behavior of a kicker. A impulse response
        of a phase linearized lowpass filter is modified Bessel function of the second kind (np.special.k0).
        The transfer function has been derived by Gerd Kotzian.
    """

    # TODO: Add 2 pi?
    def __init__(self, f_cutoff, norm_type = 'matrix_sum', norm_range = None):
        scaling = f_cutoff/c
        self.norm_range_coeff = 10
        super(self.__class__, self).__init__(scaling, norm_type, norm_range)

    def response_function(self,x):
        if x == 0:
            return 0
        else:
            return special.k0(abs(x))


class LowpassFilter(LinearProcessor):
    """ Classical first order lowpass filter (e.g. a RC filter), whose impulse response can be described as exponential
        decay.
    """
    def __init__(self, f_cutoff, norm_type = 'matrix_sum', norm_range = None):
        self.scaling = 2.*pi*f_cutoff/c
        super(self.__class__, self).__init__(self.scaling, norm_type, norm_range)

    def response_function(self,x):
        if x < 0:
            return 0
        else:
            return math.exp(-1.*x)


class Bypass(object):
    def process(self,signal, *args):
        return signal


class Weighter(object):
    """ A general class for signal weighing. A seed for the weight is a property of slices or the signal itself.
        The weight is calculated by passing the weight seed through weight_function.
    """

    def __init__(self, weight_seed, weight_normalization = None, recalculate_weight = False):
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

        self.weight_seed = weight_seed
        self.weight_normalization = weight_normalization
        self.recalculate_weight = recalculate_weight

        self.weight = None

    def weight_function(self, *args):
        pass

    def process(self,signal,slice_set):

        if (self.weight is None) or self.recalculate_weight:
            self.calculate_weight(signal,slice_set)

        return signal*self.weight

    def calculate_weight(self,signal,slice_set):
        if self.weight_seed == 'bin_length':
            bin_set = slice_set.z_bins
            self.weight = np.array([(j-i) for i, j in zip(bin_set, bin_set[1:])])
        elif self.weight_seed == 'bin_midpoint':
            bin_set = slice_set.z_bins
            self.weight = np.array([(i+j)/2 for i, j in zip(bin_set, bin_set[1:])])
        elif self.weight_seed == 'signal':
            self.weight = np.array(signal)
        else:
            self.weight = np.array(getattr(slice_set,self.weight_seed))

        self.weight = self.weight_function(self.weight)

        if self.weight_normalization == 'total_weight':
            norm_coeff = float(np.sum(self.weight))
        elif self.weight_normalization == 'average_weight':
            norm_coeff = float(np.sum(self.weight))/float(len(self.weight))
        elif self.weight_normalization == 'maximum_weight':
            norm_coeff = float(np.max(self.weight))
        elif self.weight_normalization == 'minimum_weight':
            norm_coeff = float(np.min(self.weight))
        else:
            norm_coeff = 1.

        self.weight = self.weight / norm_coeff


class ChargeWeighter(Weighter):
    """ weight signal with charge (macroparticles) of slices
    """

    def __init__(self, normalization = 'maximum_weight'):
        super(self.__class__, self).__init__('n_macroparticles_per_slice', normalization)

    def weight_function(self,weight):
        return weight


class FermiDiracInverseWeighter(Weighter):
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

    def weight_function(self,weight):
        weight = np.exp((np.absolute(weight)-self.bunch_length/2.)/float(self.bunch_decay_length))+ 1.
        weight = np.clip(weight,1.,self.maximum_weight)
        return weight


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
        pass

    def __call__(self, reader_position = None):
        # stores a reader position. During a call of a iterator goes to __iter__ function
        self.reader_position = reader_position
        return self

    def process(self,signal, *args):

        self.register.append(signal)

        if len(self.register) > self.max_reg_length:
            self.register.popleft()

        if self.in_processor_chain == True:
            temp_signal = np.zeros(len(signal))
            if len(self) > 0:
                for value in self:
                    temp_signal += value/float(len(self))
            return temp_signal


class PlainRegister(Register):
    # A plain register, which does not modify signals. The function process() returns the latest value in the register.
    # In the other cases, next() returns unmodified values.

    def __init__(self,delay, avg_length, phase_shift_per_turn, position=None, n_slices=None, in_processor_chain=True):
        super(self.__class__, self).__init__(delay, avg_length, phase_shift_per_turn, position, n_slices, in_processor_chain)

    def next(self):
        if self.n_iter_left == 0:
            raise StopIteration
        elif self.n_iter_left == -1:
            self.n_iter_left = 0
            return np.zeros(len(self.register[0]))
        elif self.reader_position is None:
            self.n_iter_left = 0
            return self.register[len(self.register)-self.delay-1]
        else:
            self.n_iter_left -= 1
            return self.register[self.n_iter_left]


class CosineSumRegister(Register):
    """ Returns register values by multiplying the values with a cosine of the betatron phase angle from the reader.
        If there are multiple values in different phases, the sum approaches a value equal to half of the displacement
        in the reader's position.

        The function process() returns a value, which is an average of the register values (after delay determined by
        the parameter avg_length)
    """
    def __init__(self,delay, avg_length, phase_shift_per_turn, position=None, n_slices=None, in_processor_chain=True):
        super(self.__class__, self).__init__(delay, avg_length, phase_shift_per_turn, position, n_slices, in_processor_chain)


    def next(self):
        if self.n_iter_left == 0:
            raise StopIteration

        elif self.n_iter_left == -1:
            self.n_iter_left = 0
            return np.zeros(len(self.register[0]))

        else:
            if self.reader_position is None:
                self.reader_position = 0

            if self.position is None:
                self.position = 0

            delta_phi = self.position - self.reader_position
            if delta_phi > pi / 2.:
                delta_phi = self.position - self.reader_position - 1. * self.phase_shift_per_turn

            delay = -1.*(len(self.register)-self.n_iter_left)
            self.n_iter_left -= 1
            return 2.*math.cos(delay*self.phase_shift_per_turn+delta_phi)*self.register[self.n_iter_left]


# class HilbertRegister(Register):
#     # A register which uses Hilber transform to calculate a betatron oscillation amplitude. The register returns in the
#     # both cases (call of process() and iteration) a single value, which is an averaged oscillation amplitudevalue.
#     # Note that avg_length must be sufficient (e.g. > X) in order to do a reliable calculation
#
#     def __init__(self,delay, avg_length, phase_shift_per_turn, position=None, n_slices=None, in_processor_chain=True):
#         super(self.__class__, self).__init__(delay, avg_length, phase_shift_per_turn, position, n_slices, in_processor_chain)
#
#     def next(self):
#         if self.n_iter_left == 0:
#             raise StopIteration
#
#         elif self.n_iter_left == -1:
#             self.n_iter_left = 0
#             return np.zeros(len(self.register[0]))
#
#         else:
#             signal = np.zeros(len(self.register[0]))
#             if len(self.register) == self.max_reg_length:
#                 temp_register = []
#                 for i, row in enumerate(self.register[0:self.avg_length]):
#                     for j,value in enumerate(row):
#                         if i == 0:
#                             temp_register.append(np.zeros(self.avg_length))
#                         temp_register[j][i] = value
#                 for i,row in  enumerate(temp_register):
#                     h = signal.hilbert(row)
#
#                     signal[i] = np.mean(np.sqrt(np.real(h[1:-1])*np.real(h[1:-1])+np.imag(h[1:-1])*np.imag(h[1:-1])))
#
#             return signal