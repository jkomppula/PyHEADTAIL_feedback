from abc import ABCMeta, abstractmethod
from scipy import signal
import copy, math
import numpy as np
from scipy.constants import c, pi
from itertools import izip, count
from processors import Register

class Resampler(object):
    def __init__(self,type, sampling_rate, sync_method):

        """ Changes a sampling rate of the signal. Assumes that either the sampling of the incoming (ADC) or
            the outgoing (DAC) signal corresponds to the sampling found from the slice_set.

        :param type: type of the conversion, i.e. 'ADC' or 'DAC'
        :param sampling_rate: Samples per second
        :param sync_method: The time range of the input signal might not correspond to an integer number of
            samples determined by sampling rate.
                'rounded': The time range of the input signal is divided to number of samples, which correspons to
                    the closest integer of samples determined by the sampling rate (defaul)
                'rising_edge': the exact value of the sampling rate is used, but there are empty space in the end
                    of the signal
                'falling_edge': the exact value of the sampling rate is used, but there are empty space in the beginning
                    of the signal
                'middle': the exact value of the sampling rate is used, but there are an equal amount of empty space
                    in the beginning and end of the signal

        """
        self._type = type
        self._sampling_rate = sampling_rate
        self._sync_method = sync_method

        self._matrix = None

    def __generate_matrix(self, z_bins_input, z_bins_output):
        self._matrix = np.zeros((len(z_bins_output)-1, len(z_bins_input)-1))
        for i, bin_in_min, bin_in_max in izip(count(), z_bins_input,z_bins_input[1:]):
            for j, bin_out_min, bin_out_max in izip(count(), z_bins_output, z_bins_output[1:]):
                out_bin_length = bin_out_max - bin_out_min
                in_bin_length = bin_in_max - bin_in_min

                self._matrix[j][i] = (self.__CDF(bin_out_max, bin_in_min, bin_in_max) -
                                      self.__CDF(bin_out_min, bin_in_min, bin_in_max)) * in_bin_length / out_bin_length

    def __generate_new_binset(self,orginal_bins, n_signal_bins):

        new_bins = None
        signal_length = (orginal_bins[-1] - orginal_bins[0]) / c

        if self._sync_method == 'round':
            if self._sampling_rate is None:
                n_z_bins = n_signal_bins
            else:
                n_z_bins = int(math.ceil(signal_length * self._sampling_rate))
            new_bins = np.linspace(orginal_bins[0], orginal_bins[-1], n_z_bins + 1)

        elif self._sync_method == 'rising_edge':
            n_z_bins = int(math.floor(signal_length * self._sampling_rate))
            max_output_z = orginal_bins[0] + float(n_z_bins) * self._sampling_rate * c
            new_bins = np.linspace(orginal_bins[0], max_output_z, n_z_bins + 1)
        elif self._sync_method == 'falling_edge':
            n_z_bins = int(math.floor(signal_length * self._sampling_rate))
            min_output_z = orginal_bins[-1] - float(n_z_bins) * self._sampling_rate * c
            new_bins = np.linspace(min_output_z, orginal_bins[-1], n_z_bins + 1)
        elif self._sync_method == 'middle':
            n_z_bins = int(math.floor(signal_length * self._sampling_rate))
            delta_z = (orginal_bins[-1] - orginal_bins[0]) - float(n_z_bins) * self._sampling_rate * c
            new_bins = np.linspace(orginal_bins[0] + delta_z / 2., orginal_bins[-1] - delta_z / 2., n_z_bins + 1)

        return new_bins

    @staticmethod
    def __CDF(x,ref_bin_from, ref_bin_to):
        if x <= ref_bin_from:
            return 0.
        elif x < ref_bin_to:
            return (x-ref_bin_from)/float(ref_bin_to-ref_bin_from)
        else:
            return 1.

    def clear_matrix(self):
        self._matrix = None

    def process(self,signal,slice_set, *args):
        z_bins_input = None
        z_bins_output = None

        if self._matrix is None:
            if self._type == 'ADC':
                z_bins_input = slice_set.z_bins
                z_bins_output = self.__generate_new_binset(slice_set.z_bins, len(signal))
            elif self._type == 'DAC':
                z_bins_input = self.__generate_new_binset(slice_set.z_bins, len(signal))
                z_bins_output = slice_set.z_bins

            self.__generate_matrix(z_bins_input, z_bins_output)

        print self._matrix.shape
        print len(signal)
        return np.dot(self._matrix, signal)


class Digitizer(object):
    def __init__(self,n_bits,input_range):

        """ Rounds signal to discrete steps determined by the number of bits.
        :param n_bitss: the signal is rounded to 2^n_bits steps
        :param input_range: the range in which 2^n_bits steps are
        """
        self._n_bits = n_bits
        self._max_integer = np.power(2,self._n_bits)-1.
        self._input_range = input_range

    def process(self, signal, *args):
        signal -= self._input_range[0]
        signal *= self._max_integer/(self._input_range[1]-self._input_range[0])
        signal = np.round(signal)

        signal[signal < 0.] = 0.
        signal[signal > self._max_integer] = 0.

        signal /= self._max_integer / (self._input_range[1] - self._input_range[0])
        signal += self._input_range[0]

        return signal


class ADC(object):
    def __init__(self,sampling_rate, n_bits = None, input_range = None, sync_method = 'round'):
        """ A model for an analog to digital converter.
        :param sampling_rate:
        :param n_bits:
        :param input_range:
        :param sync_method:
        """
        self._resampler = Resampler('ADC', sampling_rate, sync_method)

        self._digitizer = None
        if (n_bits is not None) and (input_range is not None):
            self._digitizer = Digitizer(n_bits,input_range)

    def process(self,signal,slice_set, *args):
        signal = self._resampler.process(signal,slice_set)

        if self._digitizer is not None:
            signal = self._digitizer.process(signal)

        return signal


class DAC(object):
    def __init__(self,sampling_rate, n_bits = None, output_range = None, sync_method = 'round'):
        """ A model for an digital to analog converter.
        :param sampling_rate:
        :param n_bits:
        :param input_range:
        :param sync_method:
        """
        self._resampler = Resampler('DAC', sampling_rate, sync_method)

        self._digitizer = None
        if (n_bits is not None) and (output_range is not None):
            self._digitizer = Digitizer(n_bits,output_range)

    def process(self,signal,slice_set, *args):

        if self._digitizer is not None:
            signal = self._digitizer.process(signal)

        signal = self._resampler.process(signal,slice_set)

        return signal


class DigitalFilter(object):
    def __init__(self,coefficients):
        self._coefficients = coefficients
        self.required_variables = []

    def process(self, signal, *args):
        return np.convolve(np.array(signal), np.array(self._coefficients), mode='same')


class BypassFIR(DigitalFilter):

    def __init__(self):

        coefficients = [1.]
        super(self.__class__, self).__init__(coefficients)


class FIR_Filter(DigitalFilter):

    def __init__(self,n_taps, f_cutoffs, sampling_rate):

        self._n_taps = n_taps
        self._f_cutoffs = f_cutoffs
        self._nyq = sampling_rate / 2.

        coefficients = signal.firwin(self._n_taps, self._f_cutoffs, nyq=self._nyq)

        super(self.__class__, self).__init__(coefficients)


class FIR_Register(Register):
    def __init__(self, n_taps, tune, delay, zero_idx, position, n_slices, in_processor_chain):
        self.type = 'FIR'
        self._zero_idx = zero_idx
        self._n_taps = n_taps

        super(FIR_Register, self).__init__(n_taps, tune, delay, position, n_slices, in_processor_chain)
        self.required_variables = []

    def combine(self,x1,x2,reader_position,x_to_xp = False):
        delta_phi = -1. * float(self.delay) * self.phase_shift_per_turn

        if self._zero_idx == 'middle':
            delta_phi -= float(self._n_taps/2) * self.phase_shift_per_turn

        if reader_position is not None:
            delta_position = self.position - reader_position
            delta_phi += delta_position
            if delta_position > 0:
                delta_phi -= self.phase_shift_per_turn
            if x_to_xp == True:
                delta_phi -= pi/2.

        n = self.n_iter_left

        if self._zero_idx == 'middle':
            n -= self._n_taps/2

        h = self.coeff_generator(n, delta_phi)
        h *= self._n_taps

        # print str(len(self)/2) + 'n: ' + str(n) + ' -> ' + str(h)  + ' (phi = ' + str(delta_phi) + ')'

        return np.array([h*x1[0],None])

    def coeff_generator(self, n, delta_phi):
        return 0.


class HilbertPhaseShiftRegister(FIR_Register):
    def __init__(self,n_taps, tune, delay = 0, position=None, n_slices=None, in_processor_chain=True):
        super(self.__class__, self).__init__(n_taps, tune, delay, 'middle', position, n_slices, in_processor_chain)

    def coeff_generator(self, n, delta_phi):
        h = 0.

        if n == 0:
            h = np.cos(delta_phi)
        elif n % 2 == 1:
            h = -2. * np.sin(delta_phi) / (pi * float(n))

        return h