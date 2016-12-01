import itertools
import math
import copy
from collections import deque
from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.constants import c, pi
import scipy.integrate as integrate
import scipy.special as special
from scipy import linalg
import pyximport; pyximport.install()
from cython_functions import cython_matrix_product

class Multiplication(object):
    # TODO: bin set

    __metaclass__ = ABCMeta
    """ An abstract class which multiplies the input signal by an array. The multiplier array is produced by taking
        a slice property (determined in the input parameter 'seed') and passing it through the abstract method, namely
        multiplication_function(seed).
    """
    def __init__(self, seed, normalization = None, recalculate_multiplier = False, store = False):
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

        self._seed = seed
        self._normalization = normalization
        self._recalculate_multiplier = recalculate_multiplier

        self._multiplier = None

        self.required_variables = ['z_bins']

        if self._seed not in ['bin_length','bin_midpoint','signal']:
            self.required_variables.append(self._seed)

        self._store = store

        self.input_signal = None
        self.input_bin_edges = None

        self.output_signal = None
        self.output_bin_edges = None

    @abstractmethod
    def multiplication_function(self, seed):
        pass

    def process(self,bin_edges, signal, slice_sets, phase_advance=None):

        if (self._multiplier is None) or self._recalculate_multiplier:
            self.__calculate_multiplier(signal,slice_sets)

        output_signal =  self._multiplier*signal

        if self._store:
            self.input_signal = np.copy(signal)
            self.input_bin_edges = np.copy(bin_edges)
            self.output_signal = np.copy(output_signal)
            self.output_bin_edges = np.copy(bin_edges)

        # process the signal
        return bin_edges, output_signal

    def __calculate_multiplier(self,signal,slice_sets):
        if not isinstance(slice_sets, list):
            slice_sets = [slice_sets]

        if self._multiplier is None:
            self._multiplier = np.zeros(len(signal))

        if self._seed == 'bin_length':
            start_idx = 0
            for slice_set in slice_sets:
                np.copyto(self._multiplier[start_idx:(start_idx+len(slice_set.z_bins)-1)],(slice_set.z_bins[1:]-slice_set.z_bins[:-1]))
                start_idx += (len(slice_set.z_bins)-1)

        elif self._seed == 'bin_midpoint':
            start_idx = 0
            for slice_set in slice_sets:
                np.copyto(self._multiplier[start_idx:(start_idx+len(slice_set.z_bins)-1)],(slice_set.z_bins[1:]+slice_set.z_bins[:-1])/2.)
                start_idx += (len(slice_set.z_bins)-1)

        elif self._seed == 'signal':
            np.copyto(self._multiplier,signal)

        else:
            start_idx = 0
            for slice_set in slice_sets:
                seed = getattr(slice_set,self._seed)
                np.copyto(self._multiplier[start_idx:(start_idx+len(seed))],seed)
                start_idx += len(seed)

        self._multiplier = self.multiplication_function(self._multiplier)

        if self._normalization == 'total_weight':
            norm_coeff = float(np.sum(self._multiplier))
        elif self._normalization == 'average_weight':
            norm_coeff = float(np.sum(self._multiplier))/float(len(self._multiplier))
        elif self._normalization == 'maximum_weight':
            norm_coeff = float(np.max(self._multiplier))
        elif self._normalization == 'minimum_weight':
            norm_coeff = float(np.min(self._multiplier))
        elif self._normalization == None:
            norm_coeff = 1.

        # TODO: try to figure out why this can not be written
        # self._multiplier /= norm_coeff
        self._multiplier =  self._multiplier / norm_coeff


class ChargeWeighter(Multiplication):
    """ weights signal with charge (macroparticles) of slices
    """

    def __init__(self, normalization = 'maximum_weight', store = False):
        super(self.__class__, self).__init__('n_macroparticles_per_slice', normalization,recalculate_multiplier = True,
                                             store = store)
        self.label = 'Charge weighter'

    def multiplication_function(self,weight):
        return weight


class EdgeWeighter(Multiplication):
    """ Use an inverse of the Fermi-Dirac distribution function to increase signal strength on the edges of the bunch
    """

    def __init__(self,bunch_length,bunch_decay_length,maximum_weight = 10):
        """
        :param bunch_length: estimated width of the bunch
        :param bunch_decay_length: slope of the function on the edge of the bunch. Smaller value, steeper slope.
        :param maximum_weight: maximum value of the weight
        """
        self._bunch_length = bunch_length
        self._bunch_decay_length = bunch_decay_length
        self._maximum_weight=maximum_weight
        super(self.__class__, self).__init__('bin_midpoint', 'minimum_weight')
        self.label = 'Edge weighter'

    def multiplication_function(self,weight):
        weight = np.exp((np.absolute(weight)-self._bunch_length/2.)/float(self._bunch_decay_length))+ 1.
        weight = np.clip(weight,1.,self._maximum_weight)
        return weight


class NoiseGate(Multiplication):
    """ Passes a signal which is greater/less than the threshold level.
    """

    def __init__(self,threshold, operator = 'greater', threshold_ref = 'amplitude'):

        self._threshold = threshold
        self._operator = operator
        self._threshold_ref = threshold_ref
        super(self.__class__, self).__init__('signal', None,recalculate_multiplier = True)
        self.label = 'Noise gate'

    def multiplication_function(self, seed):
        multiplier = np.zeros(len(seed))

        if self._threshold_ref == 'amplitude':
            comparable = np.abs(seed)
        elif self._threshold_ref == 'absolute':
            comparable = seed

        if self._operator == 'greater':
            multiplier[comparable > self._threshold] = 1
        elif self._operator == 'less':
            multiplier[comparable < self._threshold] = 1

        return multiplier


class MultiplicationFromFile(Multiplication):
    """ Multiplies the signal with an array, which is produced by interpolation from the loaded data. Note the seed for
        the interpolation can be any of those presented in the abstract function. E.g. a spatial weight can be
        determined by using a bin midpoint as a seed, nonlinear amplification can be modelled by using signal itself
        as a seed and etc...
    """

    def __init__(self,filename, x_axis='time', seed='bin_midpoint',normalization = None, recalculate_multiplier = False):
        super(self.__class__, self).__init__(seed, normalization, recalculate_multiplier)
        self.label = 'Multiplication from file'

        self._filename = filename
        self._x_axis = x_axis
        self._data = np.loadtxt(self._filename)
        if self._x_axis == 'time':
            self._data[:, 0] = self._data[:, 0] * c

    def multiplication_function(self, seed):
        return np.interp(seed, self._data[:, 0], self._data[:, 1])
