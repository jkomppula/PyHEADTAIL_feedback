import math
from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.constants import c, pi
import scipy.integrate as integrate
import timeit


class BunchImpulse(object):
    def __init__(self,bunch_idx, signal_bin_set, impulse_response_bin_set, impulse_response_value, impulse_parameters
                 , tot_n_bunches):

        self._bunch_idx = bunch_idx
        self._tot_n_bunches = tot_n_bunches

        self._impulse_response_value = impulse_response_value
        self._impulse_response_length = len(impulse_response_value)
        self._impulse_response_view = memoryview(self._impulse_response_value)

        self._zero_bin = impulse_parameters[0]
        print 'self._zero_bin: '+ str(self._zero_bin)
        self._bin_spacing = impulse_parameters[1]

        self._total_impulse = np.zeros(len(impulse_response_value) + len(signal_bin_set) - 2)

        self._total_impulse_length = len(self._total_impulse)
        self._total_impulse_view = memoryview(self._total_impulse)

        self._signal_edges = (signal_bin_set[0], signal_bin_set[-1])
        print 'self._signal_edges: '+ str(self._signal_edges)
        self._n_slices = len(signal_bin_set) - 1
        # TODO: Think this!
        # TODO: Add signal edges
        self._impulse_edges = (self._signal_edges[0] + impulse_response_bin_set[0],
                               self._signal_edges[1] + impulse_response_bin_set[-1])


        print 'self._impulse_edges: '+ str(self._impulse_edges)

        self._signal_length = len(signal_bin_set) - 1
        self.impulse_mem_views = []
        self.target_bunches = []
        self.signal_mem_views = [None]*tot_n_bunches

        self.signal = np.zeros(self._n_slices)
        self._signal_view = memoryview(self.signal)

        self._reset_signal = True # Flag which indicates that the signal has been ridden and it can be reset

    def build_impulse(self,signal):
        # self._total_impulse.fill(0.)
        # print 'Signal in build impulse' + str(signal)
        # print 'impulse_response_value in build impulse' + str(self._impulse_response_value)
        np.copyto(self._total_impulse,np.convolve(self._impulse_response_value,signal))
        # self._total_impulse = np.convolve(self._impulse_response_value,signal)
        # print 'total_impulse in build impulse' + str(self._total_impulse)

        # for i in enumerate(self._total_impulse):
        #     if (i>self._signal_length) and (i<(self._total_impulse_length-self._signal_length)):
        #         self._total_impulse[i] = np.sum(signal*)


        # for i,val in enumerate(signal):
        #     idx_from = i
        #     idx_to = idx_from + self._impulse_response_length
        #
        #     self._total_impulse[idx_from:idx_to] = val*self._impulse_response_value


    def check_if_target(self,target_idx,bunch_impulse_target):
        signal_edges = bunch_impulse_target.signal_edges
        impulse_edges = self._impulse_edges
        max_signal_length = bunch_impulse_target.max_signal_length
        max_impulse_length = self._total_impulse_length

        # TODO: simplify this

        if (impulse_edges[0] <= signal_edges[0]) and (impulse_edges[1] >= signal_edges[1]):
            # the signal inside the impulse
            idx_target_from = 0
            idx_target_to = max_signal_length
            idx_impulse_from = int((signal_edges[0]-impulse_edges[0])/self._bin_spacing)
            idx_impulse_to = idx_impulse_from + max_signal_length
            # self.impulse_mem_views = np.array(self._total_impulse_view[idx_impulse_from:idx_impulse_to], copy=False)
            self.impulse_mem_views.append(np.array(self._total_impulse_view[idx_impulse_from:idx_impulse_to], copy=False))
            self.target_bunches.append(target_idx)
            bunch_impulse_target.add_signal_to_total_signal(self._bunch_idx,idx_target_from,idx_target_to)

        elif (impulse_edges[0] >= signal_edges[0]) and (impulse_edges[1] <= signal_edges[1]):
            # the impulse inside the signal
            idx_target_from = int((impulse_edges[0]-signal_edges[0])/self._bin_spacing)
            idx_target_to = idx_target_from + max_impulse_length
            idx_impulse_from = 0
            idx_impulse_to = max_impulse_length
            # self.impulse_mem_views = np.array(self._total_impulse_view[idx_impulse_from:idx_impulse_to], copy=False)
            self.impulse_mem_views.append(np.array(self._total_impulse_view[idx_impulse_from:idx_impulse_to], copy=False))
            self.target_bunches.append(target_idx)
            bunch_impulse_target.add_signal_to_total_signal(self._bunch_idx,idx_target_from,idx_target_to)

        elif (impulse_edges[1] >= signal_edges[0]) and (impulse_edges[1] <= signal_edges[1]):
            # the impulse partially before the signal
            idx_target_from = 0
            idx_target_to = int((impulse_edges[1]-signal_edges[0])/self._bin_spacing)
            idx_impulse_from = self.max_signal_length - idx_target_to
            idx_impulse_to = self.max_signal_length
            # self.impulse_mem_views = np.array(self._total_impulse_view[idx_impulse_from:idx_impulse_to], copy=False)
            self.impulse_mem_views.append(np.array(self._total_impulse_view[idx_impulse_from:idx_impulse_to], copy=False))
            self.target_bunches.append(target_idx)
            bunch_impulse_target.add_signal_to_total_signal(self._bunch_idx,idx_target_from,idx_target_to)

        elif (impulse_edges[0] >= signal_edges[0]) and (impulse_edges[0] <= signal_edges[1]):
            # the impulse partially after the signal
            idx_target_from = max_signal_length - int((signal_edges[0]-impulse_edges[0])/self._bin_spacing)
            idx_target_to = max_signal_length
            idx_impulse_from = 0
            idx_impulse_to = idx_target_to - idx_target_from
            # self.impulse_mem_views = np.array(self._total_impulse_view[idx_impulse_from:idx_impulse_to], copy=False)
            self.impulse_mem_views.append(np.array(self._total_impulse_view[idx_impulse_from:idx_impulse_to], copy=False))
            self.target_bunches.append(target_idx)
            bunch_impulse_target.add_signal_to_total_signal(self._bunch_idx,idx_target_from,idx_target_to)

    def add_signal_to_total_signal(self,bunch_idx,idx_from,idx_to):
        print 'bunch_idx,idx_from,idx_to: ' + str(bunch_idx) + ' ' + str(idx_from) + ' ' + str(idx_to)
        print self.signal_mem_views
        print dir(self._signal_view)
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

        self._impulse_response_z = None
        self._impulse_response_value = None

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

        for bunch_impulse in self._bunch_impulses:
            bunch_impulse.clear_signal()

        for i,bunch_impulse in enumerate(self._bunch_impulses):
            if i< 10:
                t1 = timeit.default_timer()

            data_slot = (i*self._n_slices,(i+1)*self._n_slices)

            bunch_impulse.build_impulse(signal[data_slot[0]:data_slot[1]])
            if i< 10:
                t2 = timeit.default_timer()

            for j, impulse_mem_view in enumerate(bunch_impulse.impulse_mem_views):
                target_bunch = bunch_impulse.target_bunches[j]

                if self._bunch_impulses[target_bunch].signal_mem_views[i] is not None:
                    np.copyto(self._bunch_impulses[target_bunch].signal_mem_views[i], self._bunch_impulses[target_bunch].signal_mem_views[i] + impulse_mem_view)
                else:
                    raise ValueError('Memviews are not synchronized!')

            if i< 10:
                t3 = timeit.default_timer()
                print 'Impulse: ' + str(t2-t1) + ' and gather ' + str(t3-t2)

        self._output_signal.fill(0)

        for i,bunch_impulse in enumerate(self._bunch_impulses):
            idx_from = i * self._n_slices
            idx_to = (i + 1) * self._n_slices
            # if np.sum(bunch_impulse.signal) != 0:
            #     print 'bunch_impulse.signal: ' + str(bunch_impulse.signal)
            np.copyto(self._output_signal[idx_from:idx_to],bunch_impulse.signal)
        # print self._output_signal
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

        # calculates the bin width of the slices from the slice set of the first bunch
        self._bin_spacing = 0.
        n_values = 0
        for i, (mid_1, mid_2) in enumerate(zip(bunch_set[0].z_bins, bunch_set[0].z_bins[1:])):
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
            self._impulse_response_z.append((edge_2+edge_1)/2.)

        self._impulse_response_z = np.array(self._impulse_response_z)

        self._impulse_response_value = self.response_function(self._impulse_response_z,self._impulse_response_edges)
        print self._impulse_response_edges
        print self._impulse_response_z
        print self._impulse_response_value
        # self._signal_length_increment = max(0,self._zero_bin) + max(0,(len(self._impulse_response_value)-self._zero_bin-1))



    # +-1 bin jitter in impulse response to another bunches
    # TODO: next step:calculate off sets
    def __generate_bunch_impulses(self,bunch_set):

        self._bunch_impulses = []

        impulse_parameters = (self._zero_bin,self._bin_spacing)


        for i, bunch in enumerate(bunch_set):
            self._bunch_impulses.append(BunchImpulse(i, bunch.z_bins,self._impulse_response_z,self._impulse_response_value,impulse_parameters,len(bunch_set)))


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
