import math
from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.constants import c, pi
import scipy.integrate as integrate
import timeit

# TODO: signal extension to the harmonic sampling rate

class BunchImpulse(object):
    """
        An object, which handles data for one bunch. The variable total_impulse is the total impulse generated by
        the bunch and the total impulse (from all bunches) affecting this bunch is saved to variable signal.
    """

    def __init__(self,bunch_idx, output_signal, impulse_response, output_signal_edges, impulse_edges, bin_spacing):
        """

        :param bunch_idx:
        :param total_signal:
        :param impulse_response:
        :param total_signal_edges:
        :param impulse_edges:
        :param bin_spacing:
        """

        self._bunch_idx = bunch_idx
        self._bin_spacing = bin_spacing

        self._output_signal = output_signal
        self._output_signal_length = len(self._output_signal)
        self._output_signal_edges = output_signal_edges

        self._impulse_response = impulse_response

        self._total_impulse = np.zeros(len(self._impulse_response) + len(self._output_signal) - 1)
        self._total_impulse_length = len(self._total_impulse)
        self._total_impulse_edges = (self._output_signal_edges[0] + impulse_edges[0],
                                     self._output_signal_edges[1] + impulse_edges[1])

        self.signal_views = []
        self.impulse_views = []
        self.target_bunches = []

    def build_impulse(self,input_signal):
        """
        Build a total impulse of the bunch from the individual impulses of the slices.
        :param signal: a signal for this bunch
        """

        # self._total_impulse.fill(0.)
        # self._total_impulse += np.convolve(self._impulse_response_value,signal)
        np.copyto(self._total_impulse,np.convolve(self._impulse_response,input_signal))

    def check_if_target(self,target_idx,bunch_impulse_target):
        """
        This function checks if the impulse response of this bunch overlaps with the signal of the target bunch. In that
        case the necessary memory views for the impulse and the target are created.

        :param target_idx: a list index of the target bunch
        :param bunch_impulse_target: a bunch_impulse-object for the target bunch
        """

        signal_edges = bunch_impulse_target.signal_edges
        impulse_edges = self.impulse_edges
        max_signal_length = bunch_impulse_target.max_signal_length
        max_impulse_length = self.max_impulse_length

        if not ((impulse_edges[1] <= signal_edges[0]) or (impulse_edges[0] >= signal_edges[1])):

            # TODO: check rounding errors here
            if impulse_edges[0] <= signal_edges[0]:
                idx_target_from = 0
                idx_impulse_from = int((signal_edges[0]-impulse_edges[0])/self._bin_spacing)
            else:
                idx_target_from = int((impulse_edges[0]-signal_edges[0])/self._bin_spacing)
                idx_impulse_from = 0

            if impulse_edges[1] <= signal_edges[1]:
                idx_impulse_to = max_impulse_length
                idx_target_to = idx_target_from + idx_impulse_to - idx_impulse_from
            else:
                idx_target_to = max_signal_length
                idx_impulse_to = idx_impulse_from + idx_target_to - idx_target_from

            self.impulse_views.append(np.array(self._total_impulse[idx_impulse_from:idx_impulse_to], copy=False))
            self.target_bunches.append(target_idx)

            bunch_impulse_target.add_signal_view(self._bunch_idx,idx_target_from,idx_target_to)

    def add_signal_view(self,bunch_idx,idx_from,idx_to):
        """
        Add signal view to the
        :param bunch_idx: a list index of the bunch giving the impulse
        :param idx_from: an index from where the impulse starts
        :param idx_to: an index where to the impulse ends
        """

        while len(self.signal_views) < (bunch_idx + 1):
            self.signal_views.append(None)

        self.signal_views[bunch_idx] = np.array(self._output_signal[idx_from:idx_to], copy=False)

    @property
    def max_impulse_length(self):
        return self._total_impulse_length

    @property
    def max_signal_length(self):
        return self._output_signal_length

    @property
    def impulse_edges(self):
        return self._total_impulse_edges

    @property
    def signal_edges(self):
        return self._output_signal_edges



class Convolution(object):
    __metaclass__ = ABCMeta
    """ An abstract class for signal processors which are based on convolution.
    """

    def __init__(self, impulse_range, store):

        self._impulse_range = impulse_range

        self.required_variables = ['z_bins','mean_z']

        self._n_slices = None
        self._n_bunches = None

        self._impulse_z_bins = None
        self._impulse_mean_z = None
        self._impulse_values = None

        self._bin_spacing = None

        self._bunch_impulses = None

        self._output_signal = None

        self._store = store

        self.input_signal = None
        self.input_bin_edges = None

        self.output_signal = None
        self.output_bin_edges = None


    def process(self, bin_edges, signal, slice_sets, phase_advance=None):

        # print 'The total signal in the processor is : ' + str(signal)

        if isinstance(slice_sets, list):
            output_bin_edges, output_signal = self.process_mpi(bin_edges,signal, slice_sets)
        else:
            output_bin_edges, output_signal = self.process_normal(bin_edges,signal, [slice_sets])

        if self._store:
            self.input_signal = np.copy(signal)
            self.input_bin_edges = np.copy(bin_edges)
            self.output_signal = np.copy(output_signal)
            self.output_bin_edges = np.copy(output_bin_edges)

        # process the signal
        return output_bin_edges, output_signal

    def process_mpi(self, bin_edges, signal, bunch_set):

        if self._output_signal is None:
            self.__init_variables(bin_edges,signal,bunch_set)
        else:
            self._output_signal.fill(0.)

        for i,bunch_impulse in enumerate(self._bunch_impulses):

            signal_from = i*self._n_slices
            signal_to = (i+1)*self._n_slices
            # print 'signal[signal_from:signal_to]: ' + str(signal[signal_from:signal_to])
            bunch_impulse.build_impulse(signal[signal_from:signal_to])

            for target_bunch, impulse_view in zip(bunch_impulse.target_bunches, bunch_impulse.impulse_views):

                if self._bunch_impulses[target_bunch].signal_views[i] is not None:

                    self._bunch_impulses[target_bunch].signal_views[i] += impulse_view
                else:
                    raise ValueError('Memviews are not synchronized!')

        # print 'self._output_signal' + str(self._output_signal)

        return bin_edges, self._output_signal


    def process_normal(self,bin_edges, signal, bunch_set):
        # TODO
        pass

    def __init_variables(self,bin_edges,signal,bunch_set):

        # generates variables
        self._n_bunches - len(bunch_set)
        self._n_slices = len(bin_edges)/self._n_bunches
        # self._n_slices = len(bunch_set[0].z_bins) - 1
        self._output_signal = np.zeros(len(signal))
        self._bin_spacing = np.mean(bin_edges[0:self._n_slices,1]-bin_edges[0:self._n_slices,0])

        # self._bin_spacing = np.mean(bunch_set[0].z_bins[1:]-bunch_set[0].z_bins[:-1])

        # generates an impulse response
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

        # generates bunch impulses
        self._bunch_impulses = []

        for i,in xrange(self._n_bunches):
            idx_from = i * self._n_slices
            idx_to = (i + 1) * self._n_slices

            impulse_edges = (self._impulse_z_bins[0], self._impulse_z_bins[-1])
            signal_from = bin_edges[self._n_slices * i,0]
            signal_to = bin_edges[self._n_slices * (i + 1)-1, 0]

            signal_edges = (signal_from, signal_to)

            self._bunch_impulses.append(BunchImpulse(i,np.array(self._output_signal[idx_from:idx_to], copy=False),
                                                     self._impulse_response_value, signal_edges, impulse_edges,
                                                     self._bin_spacing))

        # checks if the impulses overlap the signals of the bunches
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
        self._norm_type = norm_type
        self._norm_range = norm_range

        super(Filter, self).__init__(impulse_range)

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

class Sinc(Filter):
    """ A nearly ideal lowpass filter, i.e. a windowed Sinc filter. The impulse response of the ideal lowpass filter
        is Sinc function, but because it is infinite length in both positive and negative time directions, it can not be
        used directly. Thus, the length of the impulse response is limited by using windowing. Properties of the filter
        depend on the width of the window and the type of the windows and must be written down. Too long window causes
        ripple to the signal in the time domain and too short window decreases the slope of the filter in the frequency
        domain. The default values are a good compromise. More details about windowing can be found from
        http://www.dspguide.com/ch16.htm and different options for the window can be visualized, for example, by using
        code in example/test 004_analog_signal_processors.ipynb
    """

    def __init__(self, f_cutoff, delay=0., window_width = 3, window_type = 'blackman' , norm_type=None, norm_range=None, bunch_spacing = None):
        """
        :param f_cutoff: a cutoff frequency of the filter
        :param delay: a delay of the filter [s]
        :param window_width: a (half) width of the window in the units of zeros of Sinc(x) [2*pi*f_c]
        :param window_type: a shape of the window, blackman or hamming
        :param norm_type: see class LinearTransform
        :param norm_range: see class LinearTransform
        """
        self.window_width = float(window_width)
        self.window_type = window_type
        impulse_range = (-1.*pi *window_width*c/(2. * pi * f_cutoff),pi *window_width*c/(2.* pi * f_cutoff))
        super(self.__class__, self).__init__( f_cutoff, impulse_range,norm_type, norm_range)

    def _impulse_response(self, x):
        if np.abs(x/pi) > self.window_width:
            return 0.
        else:
            if self.window_type == 'blackman':
                return np.sinc(x/pi)*self.blackman_window(x)
            elif self.window_type == 'hamming':
                return np.sinc(x/pi)*self.hamming_window(x)

    def blackman_window(self,x):
        return 0.42-0.5*np.cos(2.*pi*(x/pi+self.window_width)/(2.*self.window_width))\
               +0.08*np.cos(4.*pi*(x/pi+self.window_width)/(2.*self.window_width))

    def hamming_window(self, x):
        return 0.54-0.46*np.cos(2.*pi*(x/pi+self.window_width)/(2.*self.window_width))