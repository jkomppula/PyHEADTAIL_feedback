import numpy as np
from scipy.constants import c, pi
import copy, collections
# from cython_hacks import cython_matrix_product
# from scipy.interpolate import interp1d
from scipy import interpolate
from PyHEADTAIL.feedback.core import Parameters

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


    def __convert_signal(self, parameters, signal):
        if self._new_parameters is None:
            if self._sampler_type == 'original':
                self.__init_original_sampling(parameters)
            elif self._sampler_type == 'harmonic':
                self.__init_harmonic_sampling(parameters)
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
        
    
    
    
        
        
