import numpy as np
from scipy.constants import c, pi
import copy, collections
from cython_hacks import cython_matrix_product
# from scipy.interpolate import interp1d
from scipy import interpolate
from ..core import Parameters

"""
    This file contains signal processors which can be used for emulating digital signal processing in the feedback
    module. All the processors can be used separately, but digital filters assumes uniform slice spacing (bin width).
    If UniformCharge mode is used in the slicer, uniform bin width can be formed with ADC and DAC processors.

    @author Jani Komppula
    @date 16/09/2016
    @copyright CERN

"""


class HarmonicResampler(object):
    def __init__(self, multiplier=1., h_RF=1., circumference=1., sampler_type = 'harmonic',
                 data_conversion='bin_average', extra_samples = None, store_signal=False, **kwargs):
        self._multiplier = float(multiplier)
        self._h_RF = float(h_RF)
        self._circumference = float(circumference)

        self._sampler_type = sampler_type
        self._data_conversion = data_conversion
        self._extra_samples = extra_samples

        self._new_parameters = None
        self._new_signal = None
        self._old_parameters = None

        self._conversion_coeffs = None

        self._target_map = None
        self._target_signal = None

        self.extensions = ['store']
        self._store_signal = store_signal
        self.input_signal = None
        self.input_parameters = None
        self.output_signal = None
        self.output_parameters = None
        self.label = 'Harmonic resampler'


    def __init_harmonic_sampling(self, parameters):


        first_bunch = np.min(parameters['segment_midpoints'])
        last_bunch  = np.max(parameters['segment_midpoints'])
        n_bins = self._multiplier * (last_bunch - first_bunch)/(self._circumference / self._h_RF)
        n_bins = int(round(n_bins)) + 1
        if self._extra_samples is not None:
            n_bins += self._extra_samples[0]
            n_bins += self._extra_samples[1]
        bin_width = (self._circumference / self._h_RF) / self._multiplier

        z_bins = np.linspace(0,n_bins, n_bins + 1)
        z_bins *= bin_width
        z_bins += (first_bunch - 0.5 * bin_width)
        if self._extra_samples is not None:
            z_bins -= bin_width*self._extra_samples[0]

        edges = np.transpose(np.array([z_bins[:-1], z_bins[1:]]))

        self._old_parameters = parameters

        self._new_parameters = Parameters()
        self._new_parameters['signal_class'] = 2
        self._new_parameters['bin_edges'] = edges
        self._new_parameters['n_segments'] = 1
        self._new_parameters['n_bins_per_segment'] = n_bins
        self._new_parameters['segment_midpoints'] = np.mean(z_bins)
        self._new_parameters['location'] = parameters['location']
        self._new_parameters['beta'] = parameters['beta']
        self._new_parameters['original_parameters'] = parameters


        old_edges = self._old_parameters['bin_edges']
        n_old_bins = self._old_parameters['n_segments']*self._old_parameters['n_bins_per_segment']
        new_edges = self._new_parameters['bin_edges']
        n_new_bins = self._new_parameters['n_segments']*self._new_parameters['n_bins_per_segment']

        self._target_map = [0]*(n_old_bins)
        self._target_bin_counter = [0]*(n_new_bins)
        self._target_signal = np.zeros(n_new_bins+1)
        self._new_signal = np.array(self._target_signal[:-1], copy=False)

        old_bin_mids = (old_edges[:,0]+old_edges[:,1])/2.

        for i, bin_mid in enumerate(old_bin_mids):
            target_bin = -1
            for j, edges in enumerate(new_edges):
                if (edges[0] < bin_mid) and (edges[1] >= bin_mid):
                    target_bin = j

            if target_bin != -1:
                self._target_bin_counter[target_bin] += 1
            self._target_map[i] = target_bin



        if self._data_conversion == 'sum':
            self._conversion_coeffs = np.ones(n_old_bins)
        elif self._data_conversion == 'bin_average':
            self._conversion_coeffs = np.zeros(n_old_bins)

            for i, bin_idx in enumerate(self._target_map):
                if self._target_bin_counter[bin_idx] > 0:
                    self._conversion_coeffs[i] = 1./float(self._target_bin_counter[bin_idx])


        elif self._data_conversion == 'integral':
            self._conversion_coeffs = old_edges[:,1]-old_edges[:,0]
        else:
            raise ValueError('Unknown data conversion method')

    def __init_original_sampling(self,parameters):
        if 'original_parameters' in parameters:
            self._new_parameters = parameters['original_parameters']
            new_edges = self._new_parameters['bin_edges']
            old_edges = parameters['bin_edges']


            self._new_bin_mids = (new_edges[:,0]+new_edges[:,1])/2.
            self._old_bin_mids = (old_edges[:,0]+old_edges[:,1])/2.
        else:
            raise ValueError("Original parameters cannot be found from the signal parameters")

    def __init_previous_sampling(self, idx, parameters):
        if 'original_parameters' in parameters:
            self._new_parameters = parameters['previous_parameters'][idx]
            new_edges = self._new_parameters['bin_edges']
            old_edges = parameters['bin_edges']


            self._new_bin_mids = (new_edges[:,0]+new_edges[:,1])/2.
            self._old_bin_mids = (old_edges[:,0]+old_edges[:,1])/2.
        else:
            raise ValueError("Original parameters cannot be found from the signal parameters")

    def __convert_signal(self, parameters, signal):
        if self._new_parameters is None:
            if isinstance(self._sampler_type, tuple):
                if self._sampler_type[0] == 'earlier':
                    self.__init_previous_sampling(self._sampler_type[1], parameters)
                else:
                    raise ValueError("Unknown sampler type!")
            elif isinstance(self._sampler_type, basestring):
                if self._sampler_type == 'original':
                    self.__init_previous_sampling(0,parameters)
                elif self._sampler_type == 'pervious':
                    self.__init_previous_sampling(-1,parameters)
                elif self._sampler_type == 'harmonic':
                    self.__init_harmonic_sampling(parameters)
                else:
                    raise ValueError("Unknown sampler type!")
            else:
                raise ValueError("Unknown sampler type!")

        if self._sampler_type == 'original':
#            print 'len(self._old_bin_mids): ' + str(len(self._old_bin_mids))
#            print 'len(signal): ' +str(len(signal))
#            print 'len(self._old_bin_mids): ' + str(len(self._old_bin_mids))
#            print 'len(signal): ' + str(len(signal))
            tck = interpolate.splrep(self._old_bin_mids, signal, s=0)
            output_signal = interpolate.splev(self._new_bin_mids, tck, der=0)
#            print 'original self._new_signal: ' + str(self._new_signal)

        elif self._sampler_type == 'harmonic':
            output_signal = np.zeros(self._new_parameters['n_bins_per_segment'])
#            print "signal: " + str(signal)
#            print "self._target_map: " + str(self._target_map)
#            print "self._conversion_coeffs: " + str(self._conversion_coeffs)
            temp = self._conversion_coeffs*np.copy(signal)

            for i, val in zip(self._target_map, temp):
                if i > -1:
                    output_signal[i] += val
                else:
                    print "Wrong index"

#            Why this doesn't work in the mpi environment?
#            target_signal[self._target_map] += temp

            if self._target_signal[-1] != 0.:
                print self._target_map
                print "Warning: New binset doesn't cover entirely the old signal: " + str(self._target_signal[-1])

        else:
            raise ValueError("Unknown sampler type!")

        return output_signal


    def process(self, parameters, signal,slice_sets = None, *args, **kwargs):

        output_signal = self.__convert_signal(parameters, signal)

        if self._store_signal:
            self.input_signal = np.copy(signal)
            self.output_signal = np.copy(output_signal)

            self.input_parameters = copy.deepcopy(parameters)
            self.output_parameters = copy.deepcopy(self._new_parameters)

        return self._new_parameters, output_signal



class HarmonicUpSampler(object):
    def __init__(self,multiplier, pattern = None, store_signal=False):
        self._multiplier = multiplier
        self._pattern = None


    def __init_conversion(self, parameters):
        pass


class Quantizer(object):
    def __init__(self,n_bits,input_range, store_signal = False):

        """ Quantizates signal to discrete levels determined by the number of bits and input range.
        :param n_bits: the signal is quantized (rounded) to 2^n_bits levels
        :param input_range: the maximum and minimum values for the levels in the units of input signal
        """

        self._n_bits = n_bits
        self._n_steps = np.power(2,self._n_bits)-1.
        self._input_range = input_range
        self._store_signal = store_signal
        self._step_size = (self._input_range[1]-self._input_range[0])/float(self._n_steps)

        self.signal_classes = (0, 0)
        self.extensions = ['store']
        self._store_signal = store_signal
        self.input_signal = None
        self.input_parameters = None
        self.output_signal = None
        self.output_parameters = None

        self.label = 'Quantizer'

    def process(self, parameters, signal, *args, **kwargs):
        output_signal = self._step_size*np.floor(signal/self._step_size+0.5)

        output_signal[output_signal < self._input_range[0]] = self._input_range[0]
        output_signal[output_signal > self._input_range[1]] = self._input_range[1]

        if self._store_signal:
            self.input_signal = np.copy(signal)
            self.input_parameters = copy.copy(parameters)
            self.output_signal = np.copy(output_signal)
            self.output_parameters = copy.copy(parameters)

        return parameters, output_signal


class ADC(object):
    def __init__(self,multiplier, f_RF, n_bits = None, input_range = None, store_signal = False, **kwargs):
        """ A model for an analog to digital converter, which changes a length of the input signal to correspond to
            the number of slices in the PyHEADTAIL. If parameters for the quantizer are given, it quantizes also
            the input signal to discrete levels.
        :param sampling_rate: sampling rate of the ADC [Hz]
        :param n_bits: the number of bits where to input signal is quantized. If the value is None, the input signal
                is not quantizated. The default value is None.
        :param input_range: the range for for the quantizer. If the value is None, the input signal is not quantizated.
                The default value is None.
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
        self.label = 'ADC'
        self.signal_classes = (0, 1)
        self.extensions = ['store']

        h_RF=100000.
        circumference=h_RF/f_RF*c
        print circumference
        self._resampler = HarmonicResampler(multiplier, h_RF, circumference, **kwargs)

        self._digitizer = None
        if (n_bits is not None) and (input_range is not None):
            self._digitizer = Quantizer(n_bits,input_range, *kwargs)
        elif (n_bits is not None) or (input_range is not None):
            raise ValueError('Either both n_bits and input_range must have values or they must be None')

        # for storing the signal
        self._store_signal = store_signal
        self.input_signal = None
        self.input_parameters = None
        self.output_signal = None
        self.output_parameters = None

    def process(self, parameters, signal, *args, **kwargs):
        output_parameters, output_signal = self._resampler.process(parameters, signal, *args, **kwargs)

        if self._digitizer is not None:
            output_parameters, output_signal = self._digitizer.process(output_parameters, output_signal
                                                                              , *args, **kwargs)

        if self._store_signal:
            self.input_signal = np.copy(signal)
            self.input_signal_parameters = copy.copy(parameters)
            self.output_signal = np.copy(output_signal)
            self.output_parameters = copy.copy(output_parameters)

        return output_parameters, output_signal

class DAC(object):
    def __init__(self,n_bits = None, output_range = None, store_signal = False, **kwargs):
        """ A model for an digital to analog converter, which changes a length of the input signal to correspond to
            the number of slices in the PyHEADTAIL. If parameters for the quantizer are given, it quantizes also
            the input signal to discrete levels.
        :param sampling_rate: sampling rate of the ADC [Hz]
        :param n_bits: the number of bits where to input signal is quantized. If the value is None, the input signal
                is not quantizated. The default value is None.
        :param input_range: the range for for the quantizer. If the value is None, the input signal is not quantizated.
                The default value is None.
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
        self.label = 'DAC'
        self.extensions = ['store']

        self.signal_classes = (1, 0)
        self._resampler = HarmonicResampler(sampler_type = 'original')
#        self.extensions.append('bunch')
#        self.required_variables = copy.copy(self._resampler.required_variables)

        self._digitizer = None
        if (n_bits is not None) and (output_range is not None):
            self._digitizer = Quantizer(n_bits,output_range, **kwargs)
        elif (n_bits is not None) or (output_range is not None):
            raise ValueError('Either both n_bits and input_range must have values or they must be None')

        # for storing the signal
        self._store_signal = store_signal
        self.input_signal = None
        self.input_parameters = None
        self.output_signal = None
        self.output_parameters = None

    def process(self, parameters, signal, *args, **kwargs):
        output_parameters, output_signal = self._resampler.process(parameters, signal, *args, **kwargs)

        if self._digitizer is not None:
            output_parameters, output_signal = self._digitizer.process(output_parameters, output_signal,
                                                                              *args, **kwargs)

        if self._store_signal:
            self.input_signal = np.copy(signal)
            self.input_parameters = copy.copy(parameters)
            self.output_signal = np.copy(output_signal)
            self.output_parameters = copy.copy(output_parameters)

        return output_parameters, output_signal

class ToOriginalSampling(HarmonicResampler):
    def __init__(self,*args, **kwargs):
        super(self.__class__, self).__init__(sampler_type = 'original',*args, **kwargs)
        self.label = 'To the original sampling'
