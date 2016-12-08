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

# TODO: clean code here!

class LinearTransform(object):
    __metaclass__ = ABCMeta
    """ An abstract class for signal processors which are based on linear transformation. The signal is processed by
        calculating a dot product of a transfer matrix and a signal. The transfer matrix is produced with an abstract
        method, namely response_function(*args), which returns an elements of the matrix (an effect of
        the ref_bin to the bin)
    """

    def __init__(self, norm_type=None, norm_range=None, matrix_symmetry = 'none', bin_check = False,
                 bin_middle = 'bin', recalculate_always = False, store = False):
        """

        :param norm_type: Describes normalization method for the transfer matrix
            'bunch_average':    an average value over the bunch is equal to 1
            'fixed_average':    an average value over a range given in a parameter norm_range is equal to 1
            'bunch_integral':   an integral over the bunch is equal to 1
            'fixed_integral':   an integral over a fixed range given in a parameter norm_range is equal to 1
            'matrix_sum':       a sum over elements in the middle column of the matrix is equal to 1
            None:               no normalization
        :param norm_range: Normalization length in cases of self.norm_type == 'fi
        xed_length_average' or
            self.norm_type == 'fixed_length_integral'
        :param matrix_symmetry: symmetry of the matrix is used for minimizing the number of calculable elements
            in the matrix. Implemented options are:
            'none':             all elements are calculated separately
            'fully_diagonal':   all elements are identical in diagonal direction
        :param bin_check: if True, a change of the bin_set is checked every time process() is called and matrix is
            recalculated if any change is found
        :param bin_middle: defines if middle points of the bins are determined by a middle point of the bin
            (bin_middle = 'bin') or an average place of macro particles (bin_middle = 'particles')
        """

        self._norm_type = norm_type
        self._norm_range = norm_range
        self._bin_check = bin_check
        self._bin_middle = bin_middle
        self._matrix_symmetry = matrix_symmetry

        self._z_bin_set = None
        self._matrix = None

        self._recalculate_matrix = True
        self._recalculate_matrix_always = recalculate_always

        self.required_variables = ['z_bins','mean_z']

        self._store = store

        self.input_signal = None
        self.input_bin_edges = None

        self.output_signal = None
        self.output_bin_edges = None

        self.label = None



    @abstractmethod
    def response_function(self, ref_bin_mid, ref_bin_from, ref_bin_to, bin_mid, bin_from, bin_to):
        # Impulse response function of the processor
        pass

    def process(self,bin_edges, signal, slice_sets, phase_advance=None):

        if self._recalculate_matrix:

            if self._bin_middle == 'particles':
                bin_midpoints = np.array([])
                for slice_set in slice_sets:
                    bin_midpoints = np.append(bin_midpoints, slice_set.mean_z)
            elif self._bin_middle == 'bin':
                bin_midpoints = (bin_edges[:, 1] + bin_edges[:, 0]) / 2.
            else:
                raise ValueError('Unknown value for LinearTransform._bin_middle ')

            self.__generate_matrix(bin_edges,bin_midpoints)

        output_signal = np.array(cython_matrix_product(self._matrix, signal))

        if self._store:
            self.input_signal = np.copy(signal)
            self.input_bin_edges = np.copy(bin_edges)
            self.output_signal = np.copy(output_signal)
            self.output_bin_edges = np.copy(bin_edges)

        # process the signal
        return bin_edges, output_signal

        # np.dot can't be used, because it slows down the calculations in LSF by a factor of two or more
        # return np.dot(self._matrix,signal)

    def clear(self):
        self._matrix = np.array([])
        self._recalculate_matrix = True

    def print_matrix(self):
        for row in self._matrix:
            print "[",
            for element in row:
                print "{:6.3f}".format(element),
            print "]"

    def __generate_matrix(self,bin_edges, bin_midpoints):

        self._matrix = np.identity(len(bin_midpoints))

        for i, midpoint_i in enumerate(bin_midpoints):
            for j, midpoint_j in enumerate(bin_midpoints):
                    if np.isnan(self._matrix[j][i]):
                        self._matrix[j][i] = self.response_function(midpoint_i,bin_edges[i,0],bin_edges[i,1],midpoint_j,bin_edges[j,0]
                                                               ,bin_edges[j,1])

        # FIXME: This normalization doesn't work for multi bunch bin set
        if self._norm_type == 'bunch_average':
            self._norm_coeff = bin_edges[-1,1] - bin_edges[0,0]
        elif self._norm_type == 'fixed_average':
            self._norm_coeff = self._norm_range[1] - self._norm_range[0]
        elif self._norm_type == 'bunch_integral':
            center_idx = math.floor(len(bin_midpoints) / 2)
            self._norm_coeff = self.response_function(bin_midpoints[center_idx], bin_edges[center_idx,0],
                                                      bin_edges[center_idx,1], bin_midpoints[center_idx],
                                                      bin_edges[0,0], bin_edges[-1,1])
        elif self._norm_type == 'mpi_bunch_integral_RC':
            pass
            # TODO: think this affect of single bunch signal to all

            # if bunch_data is not None:
            #     center_idx = math.floor(len(bin_midpoints) / 2)
            #     self._norm_coeff = self.response_function(bin_midpoints[center_idx], bin_set[center_idx,0],
            #                                              bin_set[center_idx,1], bin_midpoints[center_idx],
            #                                              bin_set[0,0], bin_set[-1,1])
            #

        elif self._norm_type == 'fixed_integral':
            center_idx = math.floor(len(bin_midpoints) / 2)
            self._norm_coeff = self.response_function(bin_midpoints[center_idx], bin_edges[center_idx,0],
                                                      bin_edges[center_idx,1], bin_midpoints[center_idx],
                                                     self._norm_range[0], self._norm_range[-1])
        elif self._norm_type == 'max_column':
            self._norm_coeff= np.max(np.sum(self._matrix,0))

        elif self._norm_type is None:
            self._norm_coeff = 1.

        self._matrix = self._matrix / float(self._norm_coeff)

class Averager(LinearTransform):
    """ Returns a signal, which consists an average value of the input signal. A sums of the rows in the matrix
        are normalized to be one (i.e. a sum of the input signal doesn't change).
    """

    def __init__(self,norm_type = 'max_column', norm_range = None):
        super(self.__class__, self).__init__(norm_type, norm_range)
        self.label = 'Averager'

    def response_function(self, ref_bin_mid, ref_bin_from, ref_bin_to, bin_mid, bin_from, bin_to):
        return 1


class Delay(LinearTransform):
    """ Delays signal in the units of [second].
    """
    def __init__(self,delay, norm_type = None, norm_range = None,recalculate_always = False):
        self._delay = delay
        super(self.__class__, self).__init__(norm_type, norm_range, 'fully_diagonal', recalculate_always = recalculate_always)
        self.label = 'Delay'

    def response_function(self, ref_bin_mid, ref_bin_from, ref_bin_to, bin_mid, bin_from, bin_to):

        return self.__CDF(bin_to, ref_bin_from, ref_bin_to) - self.__CDF(bin_from, ref_bin_from, ref_bin_to)

    def __CDF(self,x,ref_bin_from, ref_bin_to):
        if (x-self._delay*c) <= ref_bin_from:
            return 0.
        elif (x-self._delay*c) < ref_bin_to:
            return ((x-self._delay*c)-ref_bin_from)/float(ref_bin_to-ref_bin_from)
        else:
            return 1.


class LinearTransformFromFile(LinearTransform):
    """ Interpolates matrix columns by using inpulse response data from a file. """

    def __init__(self,filename, x_axis = 'time', norm_type = 'max_column', norm_range = None):
        self._filename = filename
        self._x_axis = x_axis
        self._data = np.loadtxt(self._filename)
        if self._x_axis == 'time':
            self._data[:, 0]=self._data[:, 0]*c

        super(self.__class__, self).__init__(norm_type, norm_range)
        self.label = 'LT from file'

    def response_function(self, ref_bin_mid, ref_bin_from, ref_bin_to, bin_mid, bin_from, bin_to):
            return np.interp(bin_mid - ref_bin_mid, self._data[:, 0], self._data[:, 1])


class LtFilter(LinearTransform):
    __metaclass__ = ABCMeta
    """ A general class for (analog) filters. Impulse response of the filter must be determined by overwriting
        the function raw_impulse_response.

        This processor includes two additional properties.

    """

    def __init__(self, filter_type, filter_symmetry,f_cutoff, delay, f_cutoff_2nd, norm_type, norm_range, bunch_spacing):
        """
        :param filter_type: Options are:
                'lowpass'
                'highpass'
        :param f_cutoff: a cut-off frequency of the filter [Hz]
        :param delay: a delay in the units of seconds
        :param f_cutoff_2nd: a second cutoff frequency [Hz], which is implemented by cutting the tip of the impulse
                    response function
        :param norm_type: see class LinearTransform
        :param norm_range: see class LinearTransform
        """


        self._bunch_spacing = bunch_spacing
        self._f_cutoff = f_cutoff
        self._delay_z = delay * c
        self._filter_type = filter_type
        self._filter_symmetry = filter_symmetry

        self._impulse_response = self.__impulse_response_generator(f_cutoff_2nd)
        super(LtFilter, self).__init__(norm_type, norm_range, 'fully_diagonal')


        self._CDF_time = None
        self._CDF_value = None
        self._PDF = None


    @abstractmethod
    def raw_impulse_response(self, x):
        """ Impulse response of the filter.
        :param x: normalized time (t*2.*pi*f_c)
        :return: response at the given time
        """
        pass

    def __impulse_response_generator(self,f_cutoff_2nd):
        """ A function which generates the response function from the raw impulse response. If 2nd cut-off frequency
            is given, the value of the raw impulse response is set to constant at the time scale below that.
            The integral over the response function is normalized to value 1.
        """

        if f_cutoff_2nd is not None:
            threshold_tau = (2.*pi * self._f_cutoff) / (2.*pi * f_cutoff_2nd)
            threshold_val_neg = self.raw_impulse_response(-1.*threshold_tau)
            threshold_val_pos = self.raw_impulse_response(threshold_tau)
            integral_neg, _ = integrate.quad(self.raw_impulse_response, -100., -1.*threshold_tau)
            integral_pos, _ = integrate.quad(self.raw_impulse_response, threshold_tau, 100.)

            norm_coeff = np.abs(integral_neg + integral_pos + (threshold_val_neg + threshold_val_pos) * threshold_tau)

            def transfer_function(x):
                if np.abs(x) < threshold_tau:
                    return self.raw_impulse_response(np.sign(x)*threshold_tau) / norm_coeff
                else:
                    return self.raw_impulse_response(x) / norm_coeff
        else:
            norm_coeff, _ = integrate.quad(self.raw_impulse_response, -100., 100.)
            norm_coeff = np.abs(norm_coeff)
            def transfer_function(x):
                    return self.raw_impulse_response(x) / norm_coeff

        return transfer_function

    def response_function(self, ref_bin_mid, ref_bin_from, ref_bin_to, bin_mid, bin_from, bin_to):
        # Frequency scaling must be done by scaling integral limits, because integration by substitution doesn't work
        # with np.quad (see quad_problem.ipynbl). An ugly way, which could be fixed.

        scaling = 2. * pi * self._f_cutoff / c
        if self._bunch_spacing is None:
            temp, _ = integrate.quad(self._impulse_response, scaling * (bin_from - (ref_bin_mid + self._delay_z)),
                                     scaling * (bin_to - (ref_bin_mid + self._delay_z)))
        else:
            # FIXME: this works well in principle
            # TODO: add option to symmetric and "reverse time" filters.

            if self._CDF_time is None:

                n_taus = 10.

                if self._filter_symmetry == 'delay':
                    int_from = scaling * (- 1. * 0.9 * self._bunch_spacing * c)
                    int_to = scaling * ( 0.1 * self._bunch_spacing * c)
                    x_from = scaling * (- 1. * self._bunch_spacing * c) - n_taus
                    x_to = n_taus
                elif self._filter_symmetry == 'advance':
                    int_from = 0.
                    int_to = scaling * ( self._bunch_spacing * c)
                    x_from = -1. * n_taus
                    x_to = n_taus +  scaling * (1. * self._bunch_spacing * c)
                elif self._filter_symmetry == 'symmetric':
                    int_from =  scaling * (- 0.5 * self._bunch_spacing * c)
                    int_to = scaling * (0.5 * self._bunch_spacing * c)
                    x_from =  scaling * (- 0.5 * self._bunch_spacing * c) - n_taus
                    x_to = n_taus + scaling * (0.5 * self._bunch_spacing * c)

                else:
                    raise ValueError('Filter symmetry is not defined correctly!')

                n_points = 10000
                self._CDF_time = np.linspace(x_from, x_to, n_points)

                step_size = (x_to-x_from)/float(n_points)
                self._CDF_value = np.zeros(n_points)
                self._PDF = np.zeros(n_points)

                prev_value = self._CDF_time[0]
                cum_value = 0.


                for i, value in enumerate(self._CDF_time):
                    fun = lambda x: self._impulse_response(value - x)
                    temp, _ = integrate.quad(fun, int_from, int_to)
                    prev_value = value
                    # print temp
                    cum_value += temp*step_size
                    self._PDF[i] = temp
                    self._CDF_value[i] = cum_value
                print 'CDF Done'

        values = np.interp([scaling * (bin_from - (ref_bin_mid + self._delay_z)),
                            scaling * (bin_to - (ref_bin_mid + self._delay_z))], self._CDF_time, self._CDF_value)

        temp = values[1] - values[0]
        if ref_bin_mid == bin_mid:
            if self._filter_type == 'highpass':
                temp += 1.

        return temp

class Sinc(LtFilter):
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
        super(self.__class__, self).__init__('lowpass', 'symmetric', f_cutoff, delay, None, norm_type, norm_range, bunch_spacing)
        self.label = 'Sinc filter'

    def raw_impulse_response(self, x):
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


class Lowpass(LtFilter):
    """ Classical first order lowpass filter (e.g. a RC filter), which impulse response can be described as exponential
        decay.
        """
    def __init__(self, f_cutoff, delay=0., f_cutoff_2nd=None, norm_type=None, norm_range=None, bunch_spacing = None):
        super(self.__class__, self).__init__('lowpass','delay', f_cutoff, delay, f_cutoff_2nd, norm_type, norm_range, bunch_spacing)
        self.label = 'Lowpass filter'

    def raw_impulse_response(self, x):
        if x < 0.:
            return 0.
        else:
            return math.exp(-1. * x)

class Highpass(LtFilter):
    """The classical version of a highpass filter, which """
    def __init__(self, f_cutoff, delay=0., f_cutoff_2nd=None, norm_type=None, norm_range=None, bunch_spacing = None):
        super(self.__class__, self).__init__('highpass','advance', f_cutoff, delay, f_cutoff_2nd, norm_type, norm_range, bunch_spacing)
        self.label = 'Highpass filter'

    def raw_impulse_response(self, x):
        if x < 0.:
            return 0.
        else:
            return -1.*math.exp(-1. * x)

class PhaseLinearizedLowpass(LtFilter):
    def __init__(self, f_cutoff, delay=0., f_cutoff_2nd=None, norm_type=None, norm_range=None, bunch_spacing = None):
        super(self.__class__, self).__init__('lowpass','symmetric', f_cutoff, delay, f_cutoff_2nd, norm_type, norm_range, bunch_spacing)
        self.label = 'Phaselinearized lowpass filter'

    def raw_impulse_response(self, x):
        if x == 0.:
            return 0.
        else:
            return special.k0(abs(x))