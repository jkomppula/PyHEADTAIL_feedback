import math
from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.constants import c, pi
import scipy.integrate as integrate



class BunchImpulse(object):
    def __init__(self,bunch_idx, bin_set, impulse_response_edges, impulse_response_value, impulse_parameters
                 , tot_n_bunches):

        self._bunch_idx = bunch_idx
        self._tot_n_bunches = tot_n_bunches

        self._impulse_response_value = impulse_response_value
        self._impulse_response_length = len(impulse_response_value)
        self._impulse_response_view = memoryview(self._impulse_response_value)

        self._zero_bin = impulse_parameters[0]
        self._impulse_length_increment = impulse_parameters[1]
        self._bin_spacing = impulse_parameters[2]

        self._total_impulse = np.zeros(len(impulse_response_value)+self._impulse_length_increment)

        self._total_impulse_length = len(self._total_impulse)
        self._total_impulse_view = memoryview(self._total_impulse)

        self._signal_edges = (bin_set[0], bin_set[-1])
        self._n_slices = len(bin_set) - 1
        self._impulse_edges = (impulse_response_edges[0] + self._bin_spacing*float(min(0, self._zero_bin)),
                               impulse_response_edges[0] + self._bin_spacing*float(max(0, self._zero_bin)))

        self._signal_length = None
        self.impulse_mem_views = [None]*tot_n_bunches
        self.signal_mem_views = [None]*tot_n_bunches

        self.signal = np.zeros(self._n_slices)
        self._signal_view = None

        self._reset_signal = True # Flag which indicates that the signal has been ridden and it can be reset

    def build_impulse(self,signal):
        self._total_impulse.fill(0.)

        for i,val in enumerate(signal):
            idx_from = i + max(0,-1*self._zero_bin)
            idx_to = idx_from + self._impulse_response_length

            self._total_impulse[idx_from:idx_to] = val*self._impulse_response_value


    def check_if_target(self,target_idx,bunch_impulse_target):
        signal_edges = bunch_impulse_target.signal_edges
        impulse_edges = self.impulse_edges
        max_signal_length = bunch_impulse_target.max_signal_length
        max_impulse_length = self._total_impulse_length

        if self._bunch_idx == target_idx:
            idx_target_from = 0
            idx_target_to = max_signal_length
            idx_impulse_from = max(0,self._zero_bin)
            idx_impulse_to = idx_impulse_from+max_signal_length
            self.impulse_mem_views[target_idx] = np.array(self._total_impulse_view[idx_impulse_from:idx_impulse_to], copy=False)
            bunch_impulse_target.add_signal_to_total_signal(self._bunch_idx,idx_target_from,idx_target_to)

        elif (impulse_edges[0] < signal_edges[0]) and (impulse_edges[1] > signal_edges[1]):
            # the signal inside the impulse
            idx_target_from = 0
            idx_target_to = max_signal_length
            idx_impulse_from = int((signal_edges[0]-impulse_edges[0])/self._bin_spacing)
            idx_impulse_to = idx_impulse_from + max_signal_length
            self.impulse_mem_views[target_idx] = np.array(self._total_impulse_view[idx_impulse_from:idx_impulse_to], copy=False)
            bunch_impulse_target.add_signal_to_total_signal(self._bunch_idx,idx_target_from,idx_target_to)

        elif (impulse_edges[0] > signal_edges[0]) and (impulse_edges[1] < signal_edges[1]):
            # the impulse inside the signal
            idx_target_from = int((impulse_edges[0]-signal_edges[0])/self._bin_spacing)
            idx_target_to = idx_target_from + max_impulse_length
            idx_impulse_from = 0
            idx_impulse_to = max_impulse_length
            self.impulse_mem_views[target_idx] = np.array(self._total_impulse_view[idx_impulse_from:idx_impulse_to], copy=False)
            bunch_impulse_target.add_signal_to_total_signal(self._bunch_idx,idx_target_from,idx_target_to)

        elif (impulse_edges[1] > signal_edges[0]) and (impulse_edges[1] < signal_edges[1]):
            # the impulse partially before the signal
            idx_target_from = 0
            idx_target_to = int((impulse_edges[1]-signal_edges[0])/self._bin_spacing)
            idx_impulse_from = self.max_signal_length - idx_target_to
            idx_impulse_to = self.max_signal_length
            self.impulse_mem_views[target_idx] = np.array(self._total_impulse_view[idx_impulse_from:idx_impulse_to], copy=False)
            bunch_impulse_target.add_signal_to_total_signal(self._bunch_idx,idx_target_from,idx_target_to)

        elif (impulse_edges[0] > signal_edges[0]) and (impulse_edges[0] < signal_edges[1]):
            # the impulse partially after the signal
            idx_target_from = max_signal_length - int((signal_edges[0]-impulse_edges[0])/self._bin_spacing)
            idx_target_to = max_signal_length
            idx_impulse_from = 0
            idx_impulse_to = idx_target_to - idx_target_from
            self.impulse_mem_views[target_idx] = np.array(self._total_impulse_view[idx_impulse_from:idx_impulse_to], copy=False)
            bunch_impulse_target.add_signal_to_total_signal(self._bunch_idx,idx_target_from,idx_target_to)

    def add_signal_to_total_signal(self,bunch_idx,idx_from,idx_to):
        self.signal_mem_views[bunch_idx] = np.array(self._signal_view[idx_from:idx_to], copy=False)

    @property
    def max_signal_length(self):
        return self._signal_length

    @property
    def impulse_edges(self):
        return self._impulse_edges

    @property
    def signal_edges(self):
        return self.signal_edges

    def clear_signal(self):
        self.signal.fill(0.)



class Convolution(object):
    __metaclass__ = ABCMeta
    """ An abstract class for signal processors which are based on convolution.
    """

    def __init__(self, norm_type=None, norm_range=None):


        self._impulse_range = None
        self._symmetry = None

        self._norm_type = norm_type
        self._norm_range = norm_range

        self.required_variables = ['z_bins','mean_z']

        self._n_slices = None

        self._impulse_response_z = None
        self._impulse_response_value = None

        self._combiner = None

        self._bin_spacing = None

        self._bunch_impulses = None

        self._zero_bin = None # index of the zero bin for the impulse response (can be outside the impulse response)
        self._signal_length_increment = None

        self._output_signal = None


    def process(self, signal, slice_set, phase_advance=None, bunch_set = None):

        if bunch_set is not None:
            return self.process_mpi(signal, bunch_set)
        else:
            return self.process_normal(signal, [slice_set])

    def process_mpi(self, signal, bunch_set):

        if self._output_signal is None:
            self.__init_variables(signal,bunch_set)

        for i,bunch_impulse in enumerate(self._bunch_impulses):
            bunch_impulse.build_impulse(self, signal)

            for j, impulse_mem_view in enumerate(bunch_impulse.impulse_mem_views):
                if impulse_mem_view is not None:
                    if self._bunch_impulses[i].signal_mem_views[j] is not None:
                        self._bunch_impulses[i].signal_mem_views[j] = self._bunch_impulses[i].signal_mem_views[j] + impulse_mem_view
                    else:
                        raise ValueError('Memviews are not synchronized!')

        self._output_signal.fill(0)

        for i,bunch_impulse in enumerate(self._bunch_impulses):
            idx_from = i * self._n_slices
            idx_to = (i + 1) * self._n_slices
            self._output_signal[idx_from:idx_to] = bunch_impulse.signal

        return self._output_signal

    def _clear_signals(self):
        for bunch_impulse in enumerate(self._bunch_impulses):
            bunch_impulse.clear_signal()

    def __init_variables(self,signal,bunch_set):

        self._n_slices = bunch_set[0].n_slices

        self._output_signal = np.zeros(len(signal))

        self.__generate_impulse_response(bunch_set)
        self.__generate_bunch_impulses(bunch_set)


    def process_normal(self, signal, bunch_set):
        pass

    def __generate_impulse_response(self,bunch_set):

        # calculates the bin width of the slices from the slice set of the first bunch
        self._bin_spacing = 0.
        n_values = 0
        for i, (mid_1, mid_2) in enumerate(zip(bunch_set[0].bin_set, bunch_set[0].bin_set[1:])):
            n_values += 1
            self._bin_spacing += (mid_2-mid_1)

        self._bin_spacing = self._bin_spacing/float(n_values)

        if self._impulse_range[0] > 0.:
            temp = np.arange(-0.5*self._bin_spacing,self._impulse_range[1],self._bin_spacing)
            n_outside = len(temp[temp<self._impulse_range[0]])
            self._impulse_response_edges = temp[n_outside:]
            self._zero_bin = -1 * n_outside


        elif self._impulse_range[1] < 0.:
            temp = np.arange(-0.5*self._bin_spacing,-1.*self._impulse_range[0],self._bin_spacing)
            self._impulse_response_edges = -1.*temp[::-1]
            n_outside = len(self._impulse_response_edges[self._impulse_response_edges>self._impulse_range[1]])
            self._impulse_response_edges = self._impulse_response_edges[:-1*n_outside]
            self._zero_bin = len(temp) - 2

        elif self._impulse_range[0] == 0:
            self._impulse_response_edges = np.arange(-0.5*self._bin_spacing,self._impulse_range[1],self._bin_spacing)
            self._zero_bin = 0

        elif self._impulse_range[1] == 0:
            temp = np.arange(-0.5*self._bin_spacing,-1.*self._impulse_range[0],self._bin_spacing)
            self._impulse_response_edges = -1. * temp[::-1]
            self._zero_bin = len(self._impulse_response_edges) - 2

        else:
            temp_1 = np.arange(0.5*self._bin_spacing,-1.*self._impulse_range[0],self._bin_spacing)
            temp_1 = -1. * temp_1[::-1]
            temp_2 = np.arange(0.5*self._bin_spacing,self._impulse_range[1],self._bin_spacing)
            self._impulse_response_edges = np.append(temp_1, temp_2)
            self._zero_bin = len(temp_1) - 1

        self._impulse_response_z = []
        for edge_1, edge_2 in zip(self._impulse_response_edges,self._impulse_response_edges[1:]):
            self._impulse_response_z.append((edge_2-edge_1)/2.)

        self._impulse_response_z = np.array(self._impulse_response_z)

        self._impulse_response_value = self.response_function(self._impulse_response_z,self._impulse_response_edges)

        self._signal_length_increment = max(0,self._zero_bin) + max(0,(len(self._impulse_response_value)-self._zero_bin-1))



    # +-1 bin jitter in impulse response to another bunches
    # TODO: next step:calculate off sets
    def __generate_bunch_impulses(self,bunch_set):

        self._bunch_impulses = []

        impulse_parameters = (self._zero_bin,self._signal_length_increment,self._bin_spacing)


        for bunch in bunch_set:
            self._bunch_impulses.append(BunchImpulse(bunch.bin_set,self._impulse_response_z,self._impulse_response_value,impulse_parameters))


        for i, bunch_impulse in enumerate(self._bunch_impulses):

            for j, bunch_impulse_target in enumerate(self._bunch_impulses):
                bunch_impulse.check_if_target(i,j,bunch_impulse_target)



    @abstractmethod
    def response_function(self, impulse_response_z,impulse_response_edges):
        # Impulse response function of the processor
        pass


class Filter(Convolution):
    __metaclass__ = ABCMeta

    def __init__(self,f_cutoff,symmetry, norm_type=None, norm_range=None):

        self._f_cutoff = f_cutoff


        super(Filter, self).__init__(norm_type, norm_range)

    def response_function(self, impulse_response_z, impulse_response_edges):

        response_values = np.zeros(len(impulse_response_z))

        scaling = 2. * pi * self._f_cutoff / c

        for i, (z_from, z_to) in enumerate(zip(impulse_response_edges, impulse_response_edges[1:])):
            int_from = z_from * scaling
            int_to = z_to * scaling

            response_values[i], _ = integrate.quad(self.__impulse_response, int_from, int_to)

    def __impulse_response(self, x):
        pass


class Lowpass(Filter):
    def __init__(self,f_cutoff, norm_type, norm_range):
        super(self.__class__, self).__init__( f_cutoff, norm_type, norm_range)
    def __impulse_response(self, x):
        if x < 0.:
            return 0.
        else:
            return -1. * math.exp(-1. * x)
