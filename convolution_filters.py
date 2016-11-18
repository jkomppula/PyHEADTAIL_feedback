
from abc import ABCMeta, abstractmethod
import numpy as np



class BunchImpulse(object):
    def __init__(self,impulse_response, slice_offsets, total_impulse_length, interacting_bunches,
                 impulse_mem_view_edges, signal_mem_view_edges):
        self._impulse_response = impulse_response
        self._impulse_length = len(self._impulse_response)

        self.impulse = np.zeros(total_impulse_length)
        self._impulse_view = memoryview(self.impulse)

        self._slice_offsets = slice_offsets

        self._interacting_bunches = interacting_bunches
        self._impulse_mem_view_edges = impulse_mem_view_edges
        self._signal_mem_view_edges = signal_mem_view_edges

        self._signal_length = None
        self._impulse_mem_views = None
        self._signal_mem_views = None

        self._signal = None
        self._signal_view = None

    def build_impulse(self,signal):
        self.impulse.fill(0.)

        if self._signal_length is None:
            self._init_memory_views(len(signal))


        for offset, strength in zip(self._slice_offsets, signal):
            self.impulse[offset:(offset + self._impulse_length)] = strength * self._impulse_response


    def _init_memory_views(self,signal_length):
        self._impulse_mem_views = []
        self._signal_mem_views = []

        self._signal = np.zeros(signal_length)
        self._signal_view = memoryview(self._signal)

        for edges in self._impulse_mem_view_edges:
            self._impulse_mem_views.append(np.array(self._impulse_view[edges[0]:edges[1]],copy = False))

        for edges in self._signal_mem_view_edges:
            self._signal_mem_views.append(np.array(self._signal_view[edges[0]:edges[1]],copy = False))


    def set_signal(self,signal):
        #TODO: return and fill... how does it work?
        signal
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

        self._z_bin_set = None
        self._matrix = None

        self.required_variables = ['z_bins','mean_z']

        self._impulse_response_z = None
        self._impulse_response_value = None

        self._combiner = None

        self._bin_spacing = None

        self._bunch_impulses = None


    def process(self, signal, slice_set, phase_advance=None, bunch_set = None):

        if bunch_set is not None:
            return self.process_mpi(signal, bunch_set)
        else:
            return self.process_normal(signal, [slice_set])

    def process_mpi(self, signal, bunch_set):

        if self._impulse_response_z is None:
            self.__generate_impulse_response(bunch_set)

        if self._bunch_impulses is None:
            self.__generate_bunch_impulses(bunch_set)

        for i,bunch_impulse in enumerate(self._bunch_impulses):
            bunch_impulse.build_impulse(self,signal)

            for j,target in enumerate(bunch_impulse.target_bunches):
                self._bunch_impulses[target].add_signal(i,bunch_impulse.impulse_mem_views[j])


    def process_normal(self, signal, bunch_set):
        pass

    def __generate_impulse_response(self,bunch_set):

        # calculates the bin width of the slices from the slice set of the first bunch
        self._bin_spacing = 0.
        n_values = 0
        for i, (mid_1, mid_2) in enumerate(zip(bunch_set[0].bin_set, bunch_set[0].bin_set[1:])):
            n_values += 1
            self._bin_spacing += mid_2-mid_1

        self._bin_spacing = self._bin_spacing/float(n_values)

        self._impulse_response_edges = 

        self._impulse_response_z = np.arange(self._impulse_range[0],self._impulse_range[1],self._bin_spacing)
        self._impulse_response_value = self.response_function(self._impulse_response_z,self._bin_spacing)


    def __generate_bunch_impulses(self,bunch_set):

        self._bunch_impulses = []

        for bunch in bunch_set:
            self._bunch_impulses.append(BunchImpulse(bunch.bin_set[0],bunch.bin_set[-1])


        for i, bunch_impulse in enumerate(self._bunch_impulses):

            for j, bunch_impulse_target in enumerate(self._bunch_impulses):

                bunch_impulse.check_if_target(i,j,bunch_impulse_target)



    @abstractmethod
    def response_function(self, impulse_response_z,bin_spacing):
        # Impulse response function of the processor
        pass



