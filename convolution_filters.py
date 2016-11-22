import math
from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.constants import c, pi
import scipy.integrate as integrate
import timeit

# TODO: signal extension to the harmonic sampling rate

class BunchImpulse(object):
    """

    """

    def __init__(self,bunch_idx, signal_bin_set, impulse_response_bin_set, impulse_response_value, impulse_parameters
                 , tot_n_bunches):

        self._bunch_idx = bunch_idx
        self._tot_n_bunches = tot_n_bunches

        self._impulse_response_value = impulse_response_value
        self._impulse_response_length = len(impulse_response_value)
        self._impulse_response_view = memoryview(self._impulse_response_value)

        self._zero_bin = impulse_parameters[0]

        self._bin_spacing = impulse_parameters[1]

        self._total_impulse = np.zeros(len(impulse_response_value) + len(signal_bin_set) - 2)

        self._total_impulse_length = len(self._total_impulse)
        self._total_impulse_view = memoryview(self._total_impulse)

        self._signal_edges = (signal_bin_set[0], signal_bin_set[-1])
        self._n_slices = len(signal_bin_set) - 1

        self._impulse_edges = (self._signal_edges[0] + impulse_response_bin_set[0],
                               self._signal_edges[1] + impulse_response_bin_set[-1])

        self._signal_length = len(signal_bin_set) - 1
        self.impulse_mem_views = []
        self.target_bunches = []
        self.signal_mem_views = [None]*tot_n_bunches

        self.signal = np.zeros(self._n_slices)
        self._signal_view = memoryview(self.signal)

    def build_impulse(self,signal):
        np.copyto(self._total_impulse,np.convolve(self._impulse_response_value,signal))

    def check_if_target(self,target_idx,bunch_impulse_target):


        signal_edges = bunch_impulse_target.signal_edges
        impulse_edges = self._impulse_edges
        max_signal_length = bunch_impulse_target.max_signal_length
        max_impulse_length = self._total_impulse_length

        if not (impulse_edges[1] <= signal_edges[0]) or (impulse_edges[0] >= signal_edges[1]):

            if impulse_edges[0] <= signal_edges[0]:
                idx_target_from = 0
                # TODO: check rounding error here
                idx_impulse_from = int((signal_edges[0]-impulse_edges[0])/self._bin_spacing)
            else:
                # TODO: check rounding error here
                idx_target_from = int((impulse_edges[0]-signal_edges[0])/self._bin_spacing)
                idx_impulse_from = 0

            if impulse_edges[1] <= signal_edges[1]:
                idx_impulse_to = max_impulse_length
                idx_target_to = idx_target_from + idx_impulse_to - idx_impulse_from
            else:
                idx_target_to = max_signal_length
                idx_impulse_to = idx_impulse_from + idx_target_to - idx_target_from

            self.impulse_mem_views.append(np.array(self._total_impulse_view[idx_impulse_from:idx_impulse_to], copy=False))
            self.target_bunches.append(target_idx)
            bunch_impulse_target.add_signal_to_total_signal(self._bunch_idx,idx_target_from,idx_target_to)

    def add_signal_to_total_signal(self,bunch_idx,idx_from,idx_to):
        self.signal_mem_views[bunch_idx] = np.array(self._signal_view[idx_from:idx_to], copy=False)

    @property
    def max_impulse_length(self):
        return self._total_impulse_length

    @property
    def max_signal_length(self):
        return self._signal_length

    @property
    def impulse_edges(self):
        return self._impulse_edges

    @property
    def signal_edges(self):
        return self._signal_edges

    def clear_signal(self):
        self.signal.fill(0.)



class Convolution(object):
    __metaclass__ = ABCMeta
    """ An abstract class for signal processors which are based on convolution.
    """

    def __init__(self, impulse_range, norm_type=None, norm_range=None):


        self._impulse_range = impulse_range
        self._symmetry = None

        self._norm_type = norm_type
        self._norm_range = norm_range

        self.required_variables = ['z_bins','mean_z']

        self._n_slices = None

        self._impulse_z_bins = None
        self._impulse_mean_z = None
        self._impulse_values = None

        self._combiner = None

        self._bin_spacing = None

        self._bunch_impulses = None

        self._zero_bin = None # index of the zero bin for the impulse response (can be outside the impulse response)

        self._output_signal = None


    def process(self, signal, slice_set, phase_advance=None, bunch_set = None):

        if bunch_set is not None:
            return self.process_mpi(signal, slice_set, bunch_set)
        else:
            return self.process_normal(signal, [slice_set])

    def process_mpi(self, signal, slice_set, bunch_set):

        if self._output_signal is None:
            self.__init_variables(signal,bunch_set)

        for i,bunch_impulse in enumerate(self._bunch_impulses):

            data_slot = (i*self._n_slices,(i+1)*self._n_slices)

            bunch_impulse.build_impulse(signal[data_slot[0]:data_slot[1]])

            for target_bunch, impulse_mem_view in zip(bunch_impulse.target_bunches, bunch_impulse.impulse_mem_views):

                if self._bunch_impulses[target_bunch].signal_mem_views[i] is not None:
                    np.copyto(self._bunch_impulses[target_bunch].signal_mem_views[i], self._bunch_impulses[target_bunch].signal_mem_views[i] + impulse_mem_view)
                else:
                    raise ValueError('Memviews are not synchronized!')


        self._output_signal.fill(0)

        for i,bunch_impulse in enumerate(self._bunch_impulses):
            idx_from = i * self._n_slices
            idx_to = (i + 1) * self._n_slices
            np.copyto(self._output_signal[idx_from:idx_to],bunch_impulse.signal)
            bunch_impulse.clear_signal()

        return self._output_signal

    def _clear_signals(self):
        for bunch_impulse in enumerate(self._bunch_impulses):
            bunch_impulse.clear_signal()

    def __init_variables(self,signal,bunch_set):

        self._n_slices = len(bunch_set[0].z_bins) - 1

        self._output_signal = np.zeros(len(signal))

        self.__generate_impulse_response(bunch_set)
        self.__generate_bunch_impulses(bunch_set)


    def process_normal(self, signal, bunch_set):
        pass

    def __generate_impulse_response(self,bunch_set):

        self._bin_spacing = np.mean(bunch_set[0].z_bins[1:]-bunch_set[0].z_bins[:-1])

        if self._impulse_range[0] < -0.5*self._bin_spacing:
            temp = np.arange(0.5*self._bin_spacing,-1.*self._impulse_range[0],self._bin_spacing)
            z_bins_minus = -1.*temp[::-1]
        else:
            z_bins_minus = np.array([-0.5 * self._bin_spacing])

        if self._impulse_range[1] > 0.5*self._bin_spacing:
            z_bins_plus = np.arange(0.5*self._bin_spacing,self._impulse_range[1],self._bin_spacing)
        else:
            z_bins_plus = np.array([0.5*self._bin_spacing])

        self._impulse_z_bins = np.append(z_bins_minus,z_bins_plus)
        self._impulse_z_bins = self._impulse_z_bins[self._impulse_z_bins >= (self._impulse_range[0] - 0.5*self._bin_spacing)]
        self._impulse_z_bins = self._impulse_z_bins[self._impulse_z_bins <= (self._impulse_range[1] + 0.5*self._bin_spacing)]

        self._impulse_mean_z = (self._impulse_z_bins[1:]+self._impulse_z_bins[:-1])/2.

        self._impulse_response_value = self.response_function(self._impulse_mean_z,self._impulse_z_bins)

        # print 'self._impulse_z_bins: ' + str(self._impulse_z_bins)
        # print 'self._impulse_mean_z: ' + str(self._impulse_mean_z)
        # print 'self._impulse_response_value: ' + str(self._impulse_response_value)

    # +-1 bin jitter in impulse response to another bunches
    # TODO: next step:calculate off sets
    def __generate_bunch_impulses(self,bunch_set):

        self._bunch_impulses = []

        impulse_parameters = (self._zero_bin,self._bin_spacing)

        for i, bunch in enumerate(bunch_set):
            self._bunch_impulses.append(BunchImpulse(i, bunch.z_bins,self._impulse_mean_z,self._impulse_response_value,impulse_parameters,len(bunch_set)))


        for i, bunch_impulse in enumerate(self._bunch_impulses):

            for j, bunch_impulse_target in enumerate(self._bunch_impulses):
                bunch_impulse.check_if_target(j,bunch_impulse_target)

    @abstractmethod
    def response_function(self, impulse_response_z,impulse_response_edges):
        # Impulse response function of the processor
        pass


class Filter(Convolution):
    __metaclass__ = ABCMeta

    def __init__(self,f_cutoff,impulse_range, norm_type, norm_range):

        self._f_cutoff = f_cutoff

        super(Filter, self).__init__(impulse_range,norm_type, norm_range)

    def response_function(self, impulse_response_z, impulse_response_edges):

        response_values = np.zeros(len(impulse_response_z))

        scaling = 2. * pi * self._f_cutoff / c

        for i, (z_from, z_to) in enumerate(zip(impulse_response_edges, impulse_response_edges[1:])):
            int_from = z_from * scaling
            int_to = z_to * scaling
            if i < 10:
                print 'int_from: ' + str(int_from)
                print 'int_to: ' + str(int_to)

            response_values[i], _ = integrate.quad(self._impulse_response, int_from, int_to)

        return response_values

    @abstractmethod
    def _impulse_response(self, x):
        pass


class Lowpass(Filter):
    def __init__(self,f_cutoff, norm_type=None, norm_range=None, impulse_length = 3.):
        impulse_range = (0,impulse_length*c/(2. * pi * f_cutoff))
        super(self.__class__, self).__init__( f_cutoff, impulse_range,norm_type, norm_range)

    def _impulse_response(self, x):
        if x < 0.:
            return 0.
        else:
            return math.exp(-1. * x)
