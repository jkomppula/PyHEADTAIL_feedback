import numpy as np
from scipy.constants import c, pi
import copy
import pyximport; pyximport.install()
from cython_functions import cython_matrix_product
from scipy.interpolate import interp1d

"""
    This file contains signal processors which can be used for emulating digital signal processing in the feedback
    module. All the processors can be used separately, but digital filters assumes uniform slice spacing (bin width).
    If UniformCharge mode is used in the slicer, uniform bin width can be formed with ADC and DAC processors.

    @author Jani Komppula
    @date 16/09/2016
    @copyright CERN

"""

# TODO: smoother DAC by using interpolation
# TODO:


class Resampler(object):
    def __init__(self,sampling_type, sampling_rate = None, signal_length = None, sync_method = 'bin_mid',
                 length_rounding = 'round', data_conversion = 'bin_average', store_signal  = False):

        self._sampling_type = sampling_type
        self._sampling_rate = sampling_rate
        self._sync_method = sync_method
        self._signal_length = signal_length*c
        self._length_rounding = length_rounding
        self._data_conversion = data_conversion

        self._n_bunches = None

        self._input_bin_spacing = None
        self._input_n_slices_per_bunch = None
        self._input_z_bins = None
        self._input_bin_edges = None
        self._total_input_bin_edges = None
        self._total_input_bin_mids = None


        self._output_z_bins = None
        self._output_n_slices_per_bunch = None
        self._output_bin_spacing = c/sampling_rate
        self._output_bin_edges = None
        self._total_output_bin_edges = None
        self._total_output_bin_mids = None

        # cache for output signal
        self._output_signal = None

        self._conversion_type = None
        self._conversion_matrix = None

        # for storing the signal
        self._store_signal  = store_signal
        self.input_signal = None
        self.input_bin_edges = None
        self.output_signal = None
        self.output_bin_edges = None

        self.required_variables = ['z_bins']
        self.label = 'Resampler'


    def process(self,bin_edges,signal,slice_sets,phase_advance, ** kwargs):
        if self._conversion_type is None:
            self.__init_variables(bin_edges,slice_sets)

        if self._conversion_type == 'matrix':
            self._output_signal.fill(0.)

            for i in xrange(len(slice_sets)):
                input_from = i * self._input_n_slices_per_bunch
                input_to = (i + 1) * self._input_n_slices_per_bunch
                output_from = i * self._output_n_slices_per_bunch
                output_to = (i + 1) * self._output_n_slices_per_bunch

                np.copyto(self._output_signal[output_from:output_to],
                          np.array(cython_matrix_product(self._conversion_matrix, np.array(signal[input_from:input_to]))))
        elif self._conversion_type == 'interpolation':
            f = interp1d(self._total_input_bin_mids, signal, kind='cubic')
            self._output_signal =  f(self._total_output_bin_mids)
        else:
            raise ValueError('Unknown value in Resampler._conversion_type')

        if self._store_signal:
            self.input_signal = np.copy(signal)
            self.input_bin_edges = np.copy(bin_edges)
            self.output_signal = np.copy(self._output_signal)
            self.output_bin_edges = np.copy(self._total_output_bin_edges)

        return self._total_output_bin_edges,self._output_signal


    def __init_variables(self,bin_edges,slice_sets):
        self._n_bunches = len(slice_sets)
        self._input_n_slices_per_bunch = len(bin_edges)/self._n_bunches

        self._input_bin_edges = np.copy(bin_edges)
        self._input_z_bins = bin_edges[0:self._input_n_slices_per_bunch,0]
        self._input_z_bins = np.append(self._input_z_bins,bin_edges[(self._input_n_slices_per_bunch-1),1])
        self._input_z_bins = self._input_z_bins - np.mean(slice_sets[0].z_bins) # A re
        self._input_bin_spacing = np.mean(bin_edges[0:self._input_n_slices_per_bunch,1]-bin_edges[0:self._input_n_slices_per_bunch,0])
        self._total_input_bin_edges = np.copy(bin_edges)
        self._total_input_bin_mids = (self._total_input_bin_edges[:,0]+self._total_input_bin_edges[:,1])/2.

        if self._sampling_type == 'reconstructed':
            z_bins, n_slices_per_bunch, bin_spacing, sampling_rate, signal_length = \
                self.__reconstruct_z_bins(self._signal_length, self._sampling_rate, self._input_z_bins)
        elif self._sampling_type == 'resampled':
            z_bins, n_slices_per_bunch, bin_spacing, sampling_rate, signal_length = \
                self.__resample_z_bins(self._signal_length, self._sampling_rate, self._input_z_bins)
        elif self._sampling_type == 'original':
            z_bins = np.copy(slice_sets[0].z_bins) - np.mean(slice_sets[0].z_bins)
            n_slices_per_bunch = len(z_bins) -1
            bin_spacing = (z_bins[-1] - z_bins[0]) / float(n_slices_per_bunch)
            sampling_rate = bin_spacing/c
            signal_length = bin_spacing * n_slices_per_bunch
        else:
            raise ValueError('Unknown value in Resampler._sampling_type')

        self._output_z_bins = z_bins
        self._output_n_slices_per_bunch = n_slices_per_bunch
        self._output_bin_spacing = bin_spacing
        self._sampling_rate = sampling_rate
        self._signal_length = signal_length
        self._output_bin_edges = np.transpose(np.array([z_bins[:-1], z_bins[1:]]))

        self._total_output_bin_edges = None
        for slice_set in slice_sets:
            edges = self._output_bin_edges + np.mean(slice_set.z_bins)

            if self._total_output_bin_edges is None:
                self._total_output_bin_edges = np.copy(edges)
            else:
                self._total_output_bin_edges = np.append(self._total_output_bin_edges, edges, axis=0)

        self._total_output_bin_mids = (self._total_output_bin_edges[:,0]+self._total_output_bin_edges[:,1])/2.
        self._output_signal = np.zeros(len(self._total_output_bin_edges))

        if self._data_conversion == 'interpolation':
            self._conversion_type = 'interpolation'
        elif self._data_conversion == 'bin_sum':
            norm_coeff = 1.
            self.__contruct_value_conversion_matrix(norm_coeff)
            self._conversion_type = 'matrix'
        elif self._data_conversion == 'bin_integral':
            # weights the signal sum from difference slices by bin spacinf,
            # i.e. the time integral of the signals stays constant
            norm_coeff = self._input_bin_spacing / self._output_bin_spacing
            self.__contruct_value_conversion_matrix(norm_coeff)
            self._conversion_type = 'matrix'
        elif self._data_conversion == 'bin_average':
            # sets output bin value to an average value of input bins contributing to the output bin
            norm_coeff = 1. / min(self._output_bin_spacing / self._input_bin_spacing,
                                           float(self._input_n_slices_per_bunch))
            self.__contruct_value_conversion_matrix(norm_coeff)
            self._conversion_type = 'matrix'
        elif self._data_conversion == 'zero_bins':
            # output bin is zero if the midpoint of the input bin is not in the bin
            self.__contruct_value_conversion_matrix(None)
            self._conversion_type = 'matrix'
        else:
            raise ValueError('Unknown value for Resampler._data_normalization')

    def __reconstruct_z_bins(self,signal_length, sampling_rate, input_z_bins):

        if self._length_rounding == 'round':
            n_slices_per_bunch = np.round(signal_length * sampling_rate / c)
            signal_length = sampling_rate * float(n_slices_per_bunch) * c
        elif self._length_rounding == 'floor':
            n_slices_per_bunch = np.floor(signal_length*sampling_rate / c)
            signal_length = sampling_rate * float(n_slices_per_bunch) * c
        elif self._length_rounding == 'ceil':
            n_slices_per_bunch = np.ceil(signal_length*sampling_rate / c)
            signal_length = sampling_rate * float(n_slices_per_bunch) * c
        elif self._length_rounding == 'exact':
            n_slices_per_bunch = np.round(signal_length*sampling_rate / c)
            sampling_rate = signal_length / (c * float(n_slices_per_bunch))
        else:
            raise ValueError('Unknown value in Resampler._length_rounding')

        bin_spacing = signal_length / float(n_slices_per_bunch)

        if self._sync_method == 'rising_edge':
            z_from = input_z_bins[0]
            z_to = z_from + signal_length

        elif self._sync_method == 'falling_edge':
            z_from = input_z_bins[-1] - signal_length
            z_to = input_z_bins[-1]
        elif self._sync_method == 'middle':
            z_from = np.mean(input_z_bins) - 0.5 * signal_length
            z_to = np.mean(input_z_bins) - 0.5 * signal_length

        elif self._sync_method == 'bin_mid':
            bins_adv = (n_slices_per_bunch - 1)/ 2
            z_from = -1. * (0.5 + float(bins_adv)) * bin_spacing
            z_to = (0.5 + float(n_slices_per_bunch - bins_adv - 1)) * bin_spacing
        elif self._sync_method == 'bin_mid_advance':
            if input_z_bins[-1] > 0.5*bin_spacing:
                bins_after = np.ceil((input_z_bins[-1] - 0.5 * bin_spacing)/bin_spacing)
            else:
                bins_after = 0.
            z_from = -1. * (0.5 + float(float(n_slices_per_bunch) - bins_after - 1.)) * bin_spacing
            z_to = (bins_after+0.5) * bin_spacing
        elif self._sync_method == 'bin_mid_delay':
            if input_z_bins[0] < -0.5*bin_spacing:
                bins_adv = np.ceil((-1.*input_z_bins[0] - 0.5 * bin_spacing)/bin_spacing)
            else:
                bins_adv = 0.
            z_from = -1. * (bins_adv+0.5) * bin_spacing
            z_to = (0.5 + float(float(n_slices_per_bunch) - bins_adv - 1.)) * bin_spacing

        else:
            raise ValueError('Unknown value for Resampler._sync')

        z_bins = np.linspace(z_from, z_to, n_slices_per_bunch + 1)

        return z_bins, n_slices_per_bunch, bin_spacing, sampling_rate, signal_length

    def __resample_z_bins(self,signal_length, sampling_rate, input_z_bins):

        signal_length = input_z_bins[-1] - input_z_bins[0]

        if self._length_rounding == 'round':
            n_slices_per_bunch = np.round(signal_length * sampling_rate / c)
            signal_length = sampling_rate * float(n_slices_per_bunch) * c
        elif self._length_rounding == 'floor':
            n_slices_per_bunch = np.floor(signal_length*sampling_rate / c)
            signal_length = sampling_rate * float(n_slices_per_bunch) * c
        elif self._length_rounding == 'ceil':
            n_slices_per_bunch = np.ceil(signal_length*sampling_rate / c)
            signal_length = sampling_rate * float(n_slices_per_bunch) * c
        elif self._length_rounding == 'exact':
            n_slices_per_bunch = np.round(signal_length*sampling_rate / c)
            sampling_rate = signal_length / (c * float(n_slices_per_bunch))
        else:
            raise ValueError('Unknown value in Resampler._length_rounding')

        bin_spacing = signal_length / float(n_slices_per_bunch)

        length_difference = signal_length - (input_z_bins[-1] - input_z_bins[0])

        z_from = input_z_bins[0] - length_difference / 2.
        z_to = input_z_bins[-1] + length_difference / 2.
        z_bins = np.linspace(z_from, z_to, n_slices_per_bunch + 1)

        return z_bins, n_slices_per_bunch, bin_spacing, sampling_rate, signal_length

    def __contruct_value_conversion_matrix(self,norm_coeff):
        self._conversion_matrix = np.zeros((len(self._output_z_bins) - 1, len(self._input_z_bins) - 1))

        for i, (i_min, i_max) in enumerate(zip(self._output_z_bins, self._output_z_bins[1:])):
            for j, (j_min, j_max) in enumerate(zip(self._input_z_bins, self._input_z_bins[1:])):
                if norm_coeff is not None:
                    self._conversion_matrix[i, j] = (self._CDF(i_max, j_min, j_max) -
                                                     self._CDF(i_min, j_min, j_max)) * norm_coeff
                else:
                    if (i_min <= (j_min + j_max)/2.) and ((j_min + j_max)/2. < i_max):
                        self._conversion_matrix[i, j] = 1.

    def _CDF(self,x,ref_bin_from, ref_bin_to):
            if x <= ref_bin_from:
                return 0.
            elif x < ref_bin_to:
                return (x-ref_bin_from)/float(ref_bin_to-ref_bin_from)
            else:
                return 1.

class Quantizer(object):
    def __init__(self,n_bits,input_range, store_signal = True):

        """ Quantizates signal to discrete levels determined by the number of bits and input range.
        :param n_bits: the signal is quantized (rounded) to 2^n_bits levels
        :param input_range: the maximum and minimum values for the levels in the units of input signal
        """

        self._n_bits = n_bits
        self._n_steps = np.power(2,self._n_bits)-1.
        self._input_range = input_range
        self._store_signal = store_signal
        self._step_size = (self._input_range[1]-self._input_range[0])/float(self._n_steps)
        self.required_variables = []

        # for storing the signal
        self._store_signal = store_signal
        self.input_signal = None
        self.input_bin_edges = None
        self.output_signal = None
        self.output_bin_edges = None

    def process(self,bin_edges,signal,slice_sets,phase_advance, ** kwargs):
        output_signal = self._step_size*np.floor(signal/self._step_size+0.5)

        output_signal[output_signal < self._input_range[0]] = self._input_range[0]
        output_signal[output_signal > self._input_range[1]] = self._input_range[1]

        if self._store_signal:
            self.input_signal = np.copy(signal)
            self.input_bin_edges = np.copy(bin_edges)
            self.output_signal = np.copy(output_signal)
            self.output_bin_edges = np.copy(bin_edges)

        return output_signal


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
#         self._resampler = Resampler('reconstructed' , sampling_rate, sync_method)
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