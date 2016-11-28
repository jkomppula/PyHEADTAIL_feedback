from abc import ABCMeta, abstractmethod
from scipy import signal
import copy, math
import numpy as np
from scipy.constants import c, pi
import math
from itertools import izip, count
from processors import Register
from scipy import linalg
import pyximport; pyximport.install()
from cython_functions import cython_matrix_product

"""
    This file contains signal processors which can be used for emulating digital signal processing in the feedback
    module. All the processors can be used separately, but digital filters assumes uniform slice spacing (bin width).
    If UniformCharge mode is used in the slicer, uniform bin width can be formed with ADC and DAC processors.

    @author Jani Komppula
    @date 16/09/2016
    @copyright CERN

"""

class Resampler(object):
    def __init__(self,type, sampling_rate, sync, signal_length, data_normalization):
        self._type = type
        self._sampling_rate = sampling_rate
        self._sync = sync
        self._signal_length = signal_length*c
        self._data_normalization = data_normalization

        self._conversion_matrix = None

        self._n_bunches = None

        self._input_bin_spacing = None
        self._input_n_slices_per_bunch = None
        self._input_z_bins = None
        self._input_bin_edges = None

        self._output_z_bins = None
        self._output_n_slices_per_bunch = None
        self._output_bin_spacing = c/sampling_rate
        self._output_bin_edges = None
        self._total_output_bin_edges = None

        self._output_signal = None

        self.required_variables = ['z_bins']


    def process(self,bin_edges,signal,slice_sets,phase_advance):
        # FIXME: single bunch simulations
        # FIXME: old_binset
        if self._conversion_matrix is None:
            print 'I should build the matrix'
            self._generate_slice_sets(bin_edges,slice_sets)
            self._build_conversion_matrices()


        self._output_signal.fill(0.)

        # print 'self._conversion_matrix: ' + str(self._conversion_matrix)
        # print 'signal: ' + str(signal)

        for i in xrange(len(slice_sets)):
            input_from = i * self._input_n_slices_per_bunch
            input_to = (i + 1) * self._input_n_slices_per_bunch
            output_from = i * self._output_n_slices_per_bunch
            output_to = (i + 1) * self._output_n_slices_per_bunch

            np.copyto(self._output_signal[output_from:output_to],
                      np.array(cython_matrix_product(self._conversion_matrix, np.array(signal[input_from:input_to]))))

        return self._total_output_bin_edges,self._output_signal

    def _build_conversion_matrices(self):
        # print 'I am building the matrix'
        # print 'len(self._input_z_bins): ' + str(len(self._input_z_bins))
        # print 'len(self._output_z_bins): ' + str(len(self._output_z_bins))
        self._conversion_matrix = np.zeros((len(self._output_z_bins)-1,len(self._input_z_bins)-1))

        for i, (i_min, i_max) in enumerate(zip(self._output_z_bins, self._output_z_bins[1:])):
            for j, (j_min, j_max) in enumerate(zip(self._input_z_bins, self._input_z_bins[1:])):
                # print 'i , j: ' + str(i) + ' ' + str(j)
                self._conversion_matrix[i,j] = (self._CDF(i_max, j_min, j_max) -
                                      self._CDF(i_min, j_min, j_max)) * self._input_bin_spacing / self._output_bin_spacing


    def _generate_slice_sets(self,bin_edges,slice_sets):

        self._n_bunches = len(slice_sets)
        self._input_n_slices_per_bunch = len(bin_edges)/self._n_bunches

        self._input_bin_edges = np.copy(bin_edges)
        self._input_z_bins = bin_edges[0:self._input_n_slices_per_bunch,0]
        self._input_z_bins = np.append(self._input_z_bins,bin_edges[(self._input_n_slices_per_bunch-1),1])
        self._input_z_bins = self._input_z_bins - np.mean(self._input_z_bins)
        self._input_bin_spacing = np.mean(bin_edges[0:self._input_n_slices_per_bunch,1]-bin_edges[0:self._input_n_slices_per_bunch,0])

        if self._type == 'ADC':
            self._generate_bin_set_for_ADC(bin_edges,slice_sets)
        elif self._type == 'DAC':
            self._generate_bin_set_for_DAC(bin_edges,slice_sets)
        elif self._type == 'FrequencyMultiplier':
            self._generate_bin_set_for_frequency_multiplier(bin_edges,slice_sets)
        else:
            raise ValueError('Unknown value for Resampler._type')

        self._output_bin_edges = np.transpose(np.array([self._output_z_bins[:-1], self._output_z_bins[1:]]))

        self._total_output_bin_edges = None
        for slice_set in slice_sets:
            edges = self._output_bin_edges + np.mean(slice_set.z_bins)

            if self._total_output_bin_edges is None:
                self._total_output_bin_edges = np.copy(edges)
            else:
                self._total_output_bin_edges = np.append(self._total_output_bin_edges, edges, axis=0)

        self._output_signal = np.zeros(len(self._total_output_bin_edges))

    def _generate_bin_set_for_DAC(self,bin_edges,slice_sets):

        self._output_z_bins = np.copy(slice_sets[0].z_bins) - np.mean(slice_sets[0].z_bins)
        self._output_n_slices_per_bunch = len(self._output_z_bins) -1
        self._output_bin_spacing = (self._output_z_bins[-1] - self._output_z_bins[0]) / float(self._output_n_slices_per_bunch)



    def _generate_bin_set_for_frequency_multiplier(self,bin_edges,slice_sets):

        if isinstance(self._sampling_rate, int) and self._sampling_rate > 1:
            self._output_bin_spacing = self._input_bin_spacing/float(self._sampling_rate)

            new_bins = np.arange(0,self._input_bin_spacing-0.1*self._output_bin_spacing,self._output_bin_spacing)
            self._output_z_bins = np.array([])
            for z_bin in self._input_z_bins[:-1]:
                self._output_z_bins = np.append(self._output_z_bins,new_bins*z_bin)

            self._output_z_bins = np.append(self._output_z_bins,self._input_z_bins[-1])
            self._output_n_slices_per_bunch = len(self._output_z_bins) - 1


        elif self._sampling_rate < 1.:
            skips = int(np.round(self._sampling_rate))
            self._output_z_bins = self._input_z_bins[::skips]

            self._output_bin_spacing = np.mean(self._output_z_bins[1:]-self._output_z_bins[:-1])
            self._output_n_slices_per_bunch = len(self._output_z_bins) - 1

        else:
            raise ValueError('Unknown value for Resampler._sampling_rate')


    def _generate_bin_set_for_ADC(self,bin_edges,slice_sets):

        if isinstance(self._signal_length, float):
            self._output_n_slices_per_bunch = int(round(self._signal_length/self._output_bin_spacing))
        elif self._signal_length == 'orginal_round':
            self._output_n_slices_per_bunch = int(round((self._input_z_bins[-1]-self._input_z_bins[0])/self._output_bin_spacing))
        elif self._signal_length == 'orginal_fit':
            self._output_n_slices_per_bunch = int(round((self._input_z_bins[-1]-self._input_z_bins)/self._output_bin_spacing))
            self._output_bin_spacing = (self._input_z_bins[-1]-self._input_z_bins[0])/float(self._output_n_slices_per_bunch)
            self._sampling_rate = self._output_bin_spacing/c
        else:
            raise ValueError('Unknown value in Resampler._signal_length')

        print 'self._output_n_slices_per_bunch: ' + str(self._output_n_slices_per_bunch)

        if self._sync == 'rising_edge':
            z_from = self._input_z_bins[0]
            z_to = z_from + self._output_n_slices_per_bunch * self._output_bin_spacing

        elif self._sync == 'falling_edge':
            z_from = self._input_z_bins[-1] - self._output_n_slices_per_bunch * self._output_bin_spacing
            z_to = self._input_z_bins[-1]

        elif self._sync == 'middle':
            z_from = -0.5 * self._output_n_slices_per_bunch * self._output_bin_spacing
            z_to = -0.5 * self._output_n_slices_per_bunch * self._output_bin_spacing
        elif self._sync == 'bin_mid':
            bins_adv = (self._output_n_slices_per_bunch - 1)/ 2
            z_from = -1. * (0.5 + float(bins_adv)) * self._output_bin_spacing
            z_to = (0.5 + float(self._output_n_slices_per_bunch - bins_adv - 1)) * self._output_bin_spacing

        elif self._sync == 'bin_mid_advance':
            if self._input_z_bins[-1] > 0.5*self._output_bin_spacing:
                bins_after = np.ceil((self._input_z_bins[-1] - 0.5 * self._output_bin_spacing)/self._output_bin_spacing)
            else:
                bins_after = 0.
            z_from = -1. * (0.5 + float(float(self._output_n_slices_per_bunch) - bins_after - 1.)) * self._output_bin_spacing
            z_to = (bins_after+0.5) * self._output_bin_spacing

        elif self._sync == 'bin_mid_delay':
            if self._input_z_bins[0] > -0.5*self._output_bin_spacing:
                bins_adv = np.ceil((-1.*self._input_z_bins[0] - 0.5 * self._output_bin_spacing)/self._output_bin_spacing)
            else:
                bins_adv = 0.
            z_from = -1. * (bins_adv+0.5) * self._output_bin_spacing
            z_to = (0.5 + float(float(self._output_n_slices_per_bunch) - bins_adv - 1.)) * self._output_bin_spacing

        else:
            raise ValueError('Unknown value for Resampler._sync')


        self._output_z_bins = np.linspace(z_from, z_to, self._output_n_slices_per_bunch + 1)


    def _CDF(self,x,ref_bin_from, ref_bin_to):
            if x <= ref_bin_from:
                return 0.
            elif x < ref_bin_to:
                return (x-ref_bin_from)/float(ref_bin_to-ref_bin_from)
            else:
                return 1.

# class Resampler(object):
#     def __init__(self,type, sampling_rate, sync_method):
#
#         """ Changes a sampling rate of the signal. Assumes that either the sampling of the incoming (ADC) or
#             the outgoing (DAC) signal corresponds to the sampling found from the slice_set.
#
#         :param type: type of the conversion, i.e. 'ADC' or 'DAC'
#         :param sampling_rate: Samples per second
#         :param sync_method: The time range of the input signal might not correspond to an integer number of
#             samples determined by sampling rate.
#                 'rounded': The time range of the input signal is divided to number of samples, which correspons to
#                     the closest integer of samples determined by the sampling rate (defaul)
#                 'rising_edge': the exact value of the sampling rate is used, but there are empty space in the end
#                     of the signal
#                 'falling_edge': the exact value of the sampling rate is used, but there are empty space in the beginning
#                     of the signal
#                 'middle': the exact value of the sampling rate is used, but there are an equal amount of empty space
#                     in the beginning and end of the signal
#
#         """
#         self._type = type
#         self._sampling_rate = sampling_rate
#         self._sync_method = sync_method
#
#         self._matrix = None
#
#         self.required_variables = ['z_bins']
#
#
#     def __generate_matrix(self, z_bins_input, z_bins_output):
#         self._matrix = np.zeros((len(z_bins_output)-1, len(z_bins_input)-1))
#         for i, bin_in_min, bin_in_max in izip(count(), z_bins_input,z_bins_input[1:]):
#             for j, bin_out_min, bin_out_max in izip(count(), z_bins_output, z_bins_output[1:]):
#                 out_bin_length = bin_out_max - bin_out_min
#                 in_bin_length = bin_in_max - bin_in_min
#
#                 self._matrix[j][i] = (self.__CDF(bin_out_max, bin_in_min, bin_in_max) -
#                                       self.__CDF(bin_out_min, bin_in_min, bin_in_max)) * in_bin_length / out_bin_length
#
#     def __generate_new_binset(self,orginal_bins, n_signal_bins):
#
#         new_bins = None
#         signal_length = (orginal_bins[-1] - orginal_bins[0]) / c
#
#         if self._sync_method == 'round':
#             if self._sampling_rate is None:
#                 n_z_bins = n_signal_bins
#             else:
#                 n_z_bins = int(math.ceil(signal_length * self._sampling_rate))
#             new_bins = np.linspace(orginal_bins[0], orginal_bins[-1], n_z_bins + 1)
#
#         elif self._sync_method == 'rising_edge':
#             n_z_bins = int(math.floor(signal_length * self._sampling_rate))
#             max_output_z = orginal_bins[0] + float(n_z_bins) * self._sampling_rate * c
#             new_bins = np.linspace(orginal_bins[0], max_output_z, n_z_bins + 1)
#         elif self._sync_method == 'falling_edge':
#             n_z_bins = int(math.floor(signal_length * self._sampling_rate))
#             min_output_z = orginal_bins[-1] - float(n_z_bins) * self._sampling_rate * c
#             new_bins = np.linspace(min_output_z, orginal_bins[-1], n_z_bins + 1)
#         elif self._sync_method == 'middle':
#             n_z_bins = int(math.floor(signal_length * self._sampling_rate))
#             delta_z = (orginal_bins[-1] - orginal_bins[0]) - float(n_z_bins) * self._sampling_rate * c
#             new_bins = np.linspace(orginal_bins[0] + delta_z / 2., orginal_bins[-1] - delta_z / 2., n_z_bins + 1)
#
#         return new_bins
#
#     @staticmethod
#     def __CDF(x,ref_bin_from, ref_bin_to):
#         if x <= ref_bin_from:
#             return 0.
#         elif x < ref_bin_to:
#             return (x-ref_bin_from)/float(ref_bin_to-ref_bin_from)
#         else:
#             return 1.
#
#     def clear_matrix(self):
#         self._matrix = None
#
#     def process(self,signal,slice_set, *args):
#
#         z_bins_input = None
#         z_bins_output = None
#
#         if self._matrix is None:
#             if self._type == 'ADC':
#                 z_bins_input = slice_set.z_bins
#                 z_bins_output = self.__generate_new_binset(slice_set.z_bins, len(signal))
#             elif self._type == 'DAC':
#                 z_bins_input = self.__generate_new_binset(slice_set.z_bins, len(signal))
#                 z_bins_output = slice_set.z_bins
#
#             self.__generate_matrix(z_bins_input, z_bins_output)
#
#         signal = np.array(signal)
#         return np.array(cython_matrix_product(self._matrix, signal))
#         # np.dot can't be used, because it slows down the calculations in LSF by a factor of two or three
#         # return np.dot(self._matrix, signal)
#
#
# class Quantizer(object):
#     def __init__(self,n_bits,input_range):
#
#         """ Quantizates signal to discrete levels determined by the number of bits and input range.
#         :param n_bits: the signal is quantized (rounded) to 2^n_bits levels
#         :param input_range: the maximum and minimum values for the levels in the units of input signal
#         """
#
#         self._n_bits = n_bits
#         self._n_steps = np.power(2,self._n_bits)-1.
#         self._input_range = input_range
#         self._step_size = (self._input_range[1]-self._input_range[0])/float(self._n_steps)
#         self.required_variables = []
#
#     def process(self, signal, *args):
#         signal = self._step_size*np.floor(signal/self._step_size+0.5)
#
#         signal[signal < self._input_range[0]] = self._input_range[0]
#         signal[signal > self._input_range[1]] = self._input_range[1]
#
#         return signal
#
#
# class ADC(object):
#     def __init__(self,sampling_rate, n_bits = None, input_range = None, sync_method = 'round'):
#         """ A model for an analog to digital converter, which changes a length of the input signal to correspond to
#             the number of slices in the PyHEADTAIL. If parameters for the quantizer are given, it quantizes also
#             the input signal to discrete levels.
#         :param sampling_rate: sampling rate of the ADC [Hz]
#         :param n_bits: the number of bits where to input signal is quantized. If the value is None, the input signal
#                 is not quantizated. The default value is None.
#         :param input_range: the range for for the quantizer. If the value is None, the input signal is not quantizated.
#                 The default value is None.
#         :param sync_method: The time range of the input signal might not correspond to an integer number of
#             samples determined by sampling rate.
#                 'rounded': The time range of the input signal is divided to number of samples, which correspons to
#                     the closest integer of samples determined by the sampling rate (defaul)
#                 'rising_edge': the exact value of the sampling rate is used, but there are empty space in the end
#                     of the signal
#                 'falling_edge': the exact value of the sampling rate is used, but there are empty space in the beginning
#                     of the signal
#                 'middle': the exact value of the sampling rate is used, but there are an equal amount of empty space
#                     in the beginning and end of the signal
#         """
#         self._resampler = Resampler('ADC', sampling_rate, sync_method)
#         self.required_variables = copy.copy(self._resampler.required_variables)
#
#         self._digitizer = None
#         if (n_bits is not None) and (input_range is not None):
#             self._digitizer = Quantizer(n_bits,input_range)
#             self.required_variables += self._digitizer.required_variables
#
#     def process(self,signal,slice_set, *args):
#         signal = self._resampler.process(signal,slice_set)
#
#         if self._digitizer is not None:
#             signal = self._digitizer.process(signal)
#
#         return signal
#
#
# class DAC(object):
#     def __init__(self,sampling_rate, n_bits = None, output_range = None, sync_method = 'round'):
#         """ A model for an digital to analog converter, which changes a length of the input signal to correspond to
#             the number of slices in the PyHEADTAIL. If parameters for the quantizer are given, it quantizes also
#             the input signal to discrete levels.
#         :param sampling_rate: sampling rate of the ADC [Hz]
#         :param n_bits: the number of bits where to input signal is quantized. If the value is None, the input signal
#                 is not quantizated. The default value is None.
#         :param input_range: the range for for the quantizer. If the value is None, the input signal is not quantizated.
#                 The default value is None.
#         :param sync_method: The time range of the input signal might not correspond to an integer number of
#             samples determined by sampling rate.
#                 'rounded': The time range of the input signal is divided to number of samples, which correspons to
#                     the closest integer of samples determined by the sampling rate (defaul)
#                 'rising_edge': the exact value of the sampling rate is used, but there are empty space in the end
#                     of the signal
#                 'falling_edge': the exact value of the sampling rate is used, but there are empty space in the beginning
#                     of the signal
#                 'middle': the exact value of the sampling rate is used, but there are an equal amount of empty space
#                     in the beginning and end of the signal
#         """
#         self._resampler = Resampler('DAC', sampling_rate, sync_method)
#         self.required_variables = copy.copy(self._resampler.required_variables)
#
#         self._digitizer = None
#         if (n_bits is not None) and (output_range is not None):
#             self._digitizer = Quantizer(n_bits,output_range)
#             self.required_variables += self._digitizer.required_variables
#
#     def process(self,signal,slice_set, *args):
#
#         if self._digitizer is not None:
#             signal = self._digitizer.process(signal)
#
#         signal = self._resampler.process(signal,slice_set)
#
#         return signal
#
#
# class DigitalFilter(object):
#     def __init__(self,coefficients):
#         """ Filters the signal by convolving the signal and the input array of filter (FIR) coefficients
#         :param coefficients: A numpy array of filter (convolution) coefficients
#         """
#         self._coefficients = coefficients
#         self.required_variables = []
#
#     def process(self, signal, *args):
#         return np.convolve(np.array(signal), np.array(self._coefficients), mode='same')
#
#
# class FIR_Filter(DigitalFilter):
#     def __init__(self,n_taps, f_cutoffs, sampling_rate):
#
#         """ A digital FIR (finite impulse response) filter, which uses firwin function from SciPy library to determine
#             filter coefficients. Note that the set value of the cut-off frequency corresponds to the real cut-off
#             frequency of the filter only when length of the signal is on the same order of longer than an period of
#             the cut off frequency and sampling rate is ("significantly") higher than the cut-off frequency. In other
#             words, do not trust too much to set value of the cut-off frequency,
#
#         :param n_taps: length of the filter (number of coefficients, i.e. the filter order + 1).
#             Odd number is recommended, when
#         :param f_cutoffs: cut-off frequencies of the filter. Multiple values are allowed as explained in
#             the documentation of firwin-function in SciPy
#         :param sampling_rate: sampling rate of the ADC (or a number of slices per seconds)
#         """
#
#         self._n_taps = n_taps
#         self._f_cutoffs = f_cutoffs
#         self._nyq = sampling_rate / 2.
#
#         coefficients = signal.firwin(self._n_taps, self._f_cutoffs, nyq=self._nyq)
#
#         super(self.__class__, self).__init__(coefficients)
#
#
# class FIR_Register(Register):
#     def __init__(self, n_taps, tune, delay, zero_idx, in_processor_chain):
#         """ A general class for the register object, which uses FIR (finite impulse response) method to calculate
#             a correct signal for kick from the register values. Because the register can be used for multiple kicker
#             (in different locations), the filter coefficients are calculated in every call with
#             the function namely coeff_generator.
#
#         :param n_taps: length of the register (and length of filter)
#         :param tune: a real number value of a betatron tune (e.g. 59.28 in horizontal or 64.31 in vertical direction
#                 for LHC)
#         :param delay: a delay between storing to reading values  in turns
#         :param zero_idx: location of the zero index of the filter coeffients
#             'middle': an index of middle value in the register is 0. Values which have spend less time than that
#                     in the register have negative indexes and vice versa
#         :param in_processor_chain: if True, process() returns a signal, if False saves computing time
#         """
#         self.combination = 'individual'
#         # self.combination = 'combined'
#         self._zero_idx = zero_idx
#         self._n_taps = n_taps
#
#         super(FIR_Register, self).__init__(n_taps, tune, delay, in_processor_chain)
#         self.required_variables = []
#
#     def combine(self,x1,x2,reader_phase_advance,x_to_xp = False):
#         delta_phi = -1. * float(self._delay) * self._phase_shift_per_turn
#
#         if self._zero_idx == 'middle':
#             delta_phi -= float(self._n_taps/2) * self._phase_shift_per_turn
#
#         if reader_phase_advance is not None:
#             delta_position = self._phase_advance - reader_phase_advance
#             delta_phi += delta_position
#             if delta_position > 0:
#                 delta_phi -= self._phase_shift_per_turn
#             if x_to_xp == True:
#                 delta_phi -= pi/2.
#
#         n = self._n_iter_left
#
#         if self._zero_idx == 'middle':
#             n -= self._n_taps/2
#         # print delta_phi
#         h = self.coeff_generator(n, delta_phi)
#         h *= self._n_taps
#
#         # print str(len(self)/2) + 'n: ' + str(n) + ' -> ' + str(h)  + ' (phi = ' + str(delta_phi) + ') from ' + str(self._phase_advance) + ' to ' + str(reader_phase_advance)
#
#         return h*x1[0]
#
#     def coeff_generator(self, n, delta_phi):
#         """ Calculates filter coefficients
#         :param n: index of the value
#         :param delta_phi: total phase advance to the kicker for the value which index is 0
#         :return: filter coefficient h
#         """
#         return 0.
#
#
# class HilbertPhaseShiftRegister(FIR_Register):
#     """ A register used in some damper systems at CERN. The correct signal is calculated by using FIR phase shifter,
#     which is based on the Hilbert transform. It is recommended to use odd number of taps (e.g. 7) """
#
#     def __init__(self,n_taps, tune, delay = 0, in_processor_chain=True):
#         super(self.__class__, self).__init__(n_taps, tune, delay, 'middle', in_processor_chain)
#
#     def coeff_generator(self, n, delta_phi):
#         h = 0.
#
#         if n == 0:
#             h = np.cos(delta_phi)
#         elif n % 2 == 1:
#             h = -2. * np.sin(delta_phi) / (pi * float(n))
#
#         return h