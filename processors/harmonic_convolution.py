import math, copy
from abc import ABCMeta, abstractmethod
import numpy as np
from scipy import signal
from scipy.constants import c, pi
import scipy.integrate as integrate
import scipy.special as special
from scipy.interpolate import UnivariateSpline

# TODO: Delay
# TODO: Jitter



class TurnConvolution(object):
    __metaclass__ = ABCMeta
    
    def __init__(self, impulse_range, store_signal=False):
        self._impulse_range = impulse_range
        
        self._coefficients = None
        
        self.extensions = ['store']
        self._store_signal = store_signal

        self.input_signal = None
        self.input_parameters = None

        self.output_signal = None
        self.output_parameters = None
        
    
    def __generate_coefficients(self,parameters):
        if self._impulse_range is not None:
            edges = parameters['bin_edges']        
            
            bin_width = np.mean(edges[:,1]-edges[:,0])
            response_from = self._impulse_range[0]
            response_to = self._impulse_range[1]
    
            min_coeff_length = max(np.abs(response_from),np.abs(response_to))
            
            n_bins_per_side = np.ceil((min_coeff_length-0.5*bin_width)/bin_width)
            
            z_bins_plus = np.linspace(bin_width/2.,bin_width/2.+n_bins_per_side*bin_width,n_bins_per_side)
            z_bins_minus = -1. * z_bins_plus[::-1]
            z_bins = np.append(z_bins_minus, z_bins_plus)
            
            impulse_bin_edges = np.transpose(np.array([z_bins[:-1], z_bins[1:]]))
            impulse_bin_mids = (impulse_bin_edges[:,0]+impulse_bin_edges[:,1])/2.
            
        else:
            impulse_bin_edges = None
            impulse_bin_mids = None
        
        self._coefficients = self.calculate_response(impulse_bin_mids,impulse_bin_edges,parameters)
#        print 'impulse_bin_mids: ' + str(impulse_bin_mids)
#        print 'impulse_bin_edges: ' + str(impulse_bin_edges)
        # print 'self._coefficients: ' + str(self._coefficients)
    
    @abstractmethod
    def calculate_response(self, bin_mids, bin_edges, parameters):
        pass


    def process(self,parameters, signal, **kwargs):
        
        if self._coefficients is None:
            self.__generate_coefficients(parameters)
        
                
        
        raw_output = np.convolve(np.array(signal), np.array(self._coefficients), mode='full')
        i_from = (len(self._coefficients)-1)/2
       #  print self.label + str(len(self._coefficients)) + ' -> i_from: ' + str(i_from)
        i_to = i_from+len(signal)
        output_signal = np.array(raw_output[i_from:i_to],copy=False)
        
        if self._store_signal:
            self.input_signal = np.copy(signal)
            self.output_signal = np.copy(output_signal)
            
            self.input_parameters = copy.deepcopy(parameters)
            self.output_parameters = copy.deepcopy(parameters)
        
        return parameters, output_signal
        
class FIRFilter(TurnConvolution):
    
    def __init__(self,coefficients, zero_tap = None, **kwargs):

        if zero_tap is not None:
            extra_zeros = 2* (zero_tap - len(coefficients)/2)
            
            if extra_zeros >= 0:
                self._input_coefficients = np.append(coefficients, np.array([0.]*extra_zeros))
            else:
                self._input_coefficients = np.append(np.array([0.]*(-1*extra_zeros)),coefficients)
                
            print 'zero_tap: ' + str(self._input_coefficients[len(self._input_coefficients)/2])
        else:    
            self._input_coefficients = coefficients
                
            
            
        
        super(FIRFilter, self).__init__(None, **kwargs)
        self.label = 'FIR filter'
        
    def calculate_response(self, impulse_bin_mids, impulse_bin_edges, parameters):
        
        return self._input_coefficients


class ConvolutionFilter(TurnConvolution):
    __metaclass__ = ABCMeta

    def __init__(self,scaling,impulse_range,zero_bin_value = None, tip_cut_width=None, normalization=None, norm_range=None, **kwargs):

        self._scaling = scaling
        self._normalization = normalization
        self._norm_range = norm_range
        self._zero_bin_value = zero_bin_value
        super(ConvolutionFilter, self).__init__(impulse_range, **kwargs)

        # NOTE: is the tip cut needed? How to work with the sharp tips of the ideal filters?
        if (self._normalization is None) and (tip_cut_width is not None):
            self._normalization = 'integral'
        self._impulse_response = self._impulse_response_generator(tip_cut_width)

    def calculate_response(self, impulse_bin_mids, impulse_bin_edges, parameters):

        # FIXME: take into account the symmetry of the impulse response

        impulse_values = np.zeros(len(impulse_bin_mids))

        for i, edges in enumerate(impulse_bin_edges):
            integral_from = edges[0] * self._scaling
            integral_to = edges[1] * self._scaling

            impulse_values[i], _ = integrate.quad(self._impulse_response, integral_from, integral_to)

        if self._normalization is None:
            pass
        elif isinstance(self._normalization, float):
            impulse_values = impulse_values/self._normalization
        elif isinstance(self._normalization, tuple):
            if self._normalization[0] == 'bunch_by_bunch':
                bunch_spacing = self._normalization[1] * c

                bunch_locations = np.array([])
                if (impulse_bin_edges[0,0] < 0):
                    bunch_locations = np.append(bunch_locations, -1.*np.arange(0.,-1.*impulse_bin_edges[0,0],bunch_spacing))
                if (impulse_bin_edges[-1,1] > 0):
                    bunch_locations = np.append(bunch_locations, np.arange(0.,impulse_bin_edges[-1,1],bunch_spacing))

                bunch_locations = np.unique(bunch_locations)

                min_mask = (bunch_locations >= impulse_bin_edges[0,0])
                max_mask = (bunch_locations <= impulse_bin_edges[-1,1])

                bunch_locations = bunch_locations[min_mask*max_mask]

                total_sum = 0.

                # TODO: check, which is the best way to calculate the normalization coefficient
                total_sum = np.sum(np.interp([bunch_locations], impulse_bin_mids, impulse_values))
#                for location in bunch_locations:
#                    min_mask = (impulse_bin_mids > (location - bunch_length/2.))
#                    max_mask = (impulse_bin_mids < (location + bunch_length/2.))
#
#                    total_sum += np.mean(impulse_values[min_mask*max_mask])

                impulse_values = impulse_values/total_sum

            else:
                raise ValueError('Unknown normalization method')

        elif self._normalization == 'max':
            impulse_values = impulse_values/np.max(impulse_values)
        elif self._normalization == 'min':
            impulse_values = impulse_values/np.min(impulse_values)
        elif self._normalization == 'average':
            impulse_values = impulse_values/np.abs(np.mean(impulse_values))
        elif self._normalization == 'sum':
            # TODO: check naming, this is not a sum, but an integral?
            impulse_values = impulse_values/np.abs(np.sum(impulse_values))
        elif self._normalization == 'integral':
            bin_widths = impulse_bin_edges[:,1]-impulse_bin_edges[:,0]
            impulse_values = impulse_values / np.abs(np.sum(impulse_values*bin_widths))
        else:
            raise ValueError('Unknown normalization method')

        if self._zero_bin_value is not None:
            for i, edges in enumerate(impulse_bin_edges):
                if (edges[0] <= 0.) and (0. < edges[1]):
                    impulse_values[i] = impulse_values[i] + self._zero_bin_value

        return impulse_values

    @abstractmethod
    def _raw_impulse_response(self, x):
        """ Impulse response of the filter.
        :param x: normalized time (t*2.*pi*f_c)
        :return: response at the given time
        """
        pass

    def _impulse_response_generator(self,tip_cut_width):
        """ A function which generates the response function from the raw impulse response. If 2nd cut-off frequency
            is given, the value of the raw impulse response is set to constant at the time scale below that.
            The integral over the response function is normalized to value 1.
        """

        if tip_cut_width is not None:
            def transfer_function(x):
                if np.abs(x) < tip_cut_width:
                    return self._raw_impulse_response(np.sign(x)*tip_cut_width)
                else:
                    return self._raw_impulse_response(x)
        else:
            def transfer_function(x):
                    return self._raw_impulse_response(x)

        return transfer_function



class Lowpass(ConvolutionFilter):
    def __init__(self,f_cutoff, impulse_length = 5., f_cutoff_2nd = None, normalization='sum', **kwargs):
        scaling = 2. * pi * f_cutoff / c
        impulse_range = (0, impulse_length/scaling)

        if f_cutoff_2nd is not None:
            tip_cut_width = f_cutoff / f_cutoff_2nd
        else:
            tip_cut_width = None

        super(self.__class__, self).__init__( scaling, impulse_range, tip_cut_width = tip_cut_width, normalization=normalization, **kwargs)
        self.label = 'Lowpass filter'

    def _raw_impulse_response(self, x):
        if x < 0.:
            return 0.
        else:
            return math.exp(-1. * x)

class Highpass(ConvolutionFilter):
    def __init__(self,f_cutoff, impulse_length = 5., f_cutoff_2nd = None, normalization='sum', **kwargs):
        scaling = 2. * pi * f_cutoff / c
        impulse_range = (0, impulse_length/scaling)

        if f_cutoff_2nd is not None:
            tip_cut_width = f_cutoff / f_cutoff_2nd
        else:
            tip_cut_width = None

        super(self.__class__, self).__init__( scaling, impulse_range, zero_bin_value= 1., tip_cut_width = tip_cut_width, normalization=normalization, **kwargs)
        self.label = 'Highpass filter'

    def _raw_impulse_response(self, x):
        if x < 0.:
            return 0.
        else:
            return -1.* math.exp(-1. * x)

class PhaseLinearizedLowpass(ConvolutionFilter):
    def __init__(self, f_cutoff, impulse_length = 5., f_cutoff_2nd = None, normalization='sum', **kwargs):
        scaling = 2. * pi * f_cutoff / c
        impulse_range = (-1.*impulse_length/scaling, impulse_length/scaling)

        if f_cutoff_2nd is not None:
            tip_cut_width = f_cutoff / f_cutoff_2nd
        else:
            tip_cut_width = None

        super(self.__class__, self).__init__( scaling, impulse_range, tip_cut_width = tip_cut_width, normalization=normalization, **kwargs)
        self.label = 'Phaselinearized lowpass filter'

    def _raw_impulse_response(self, x):
        if x == 0.:
            return 0.
        else:
            return special.k0(abs(x))


class GaussianLowpass(ConvolutionFilter):
    def __init__(self, f_cutoff, impulse_length = 5., normalization='sum', **kwargs):
        scaling = 2. * pi * f_cutoff / c
        impulse_range = (-1.*impulse_length/scaling, impulse_length/scaling)


        tip_cut_width = None

        super(self.__class__, self).__init__( scaling, impulse_range, tip_cut_width = tip_cut_width, normalization=normalization, **kwargs)
        self.label = 'Gaussian lowpass filter'

    def _raw_impulse_response(self, x):
        return np.exp(-x ** 2. / 2.) / np.sqrt(2. * pi)


class Sinc(ConvolutionFilter):
    """ A nearly ideal lowpass filter, i.e. a windowed Sinc filter. The impulse response of the ideal lowpass filter
        is Sinc function, but because it is infinite length in both positive and negative time directions, it can not be
        used directly. Thus, the length of the impulse response is limited by using windowing. Properties of the filter
        depend on the width of the window and the type of the windows and must be written down. Too long window causes
        ripple to the signal in the time domain and too short window decreases the slope of the filter in the frequency
        domain. The default values are a good compromise. More details about windowing can be found from
        http://www.dspguide.com/ch16.htm and different options for the window can be visualized, for example, by using
        code in example/test 004_analog_signal_processors.ipynb
    """

    def __init__(self, f_cutoff, window_width = 3, window_type = 'blackman', normalization='sum', **kwargs):
        """
        :param f_cutoff: a cutoff frequency of the filter
        :param delay: a delay of the filter [s]
        :param window_width: a (half) width of the window in the units of zeros of Sinc(x) [2*pi*f_c]
        :param window_type: a shape of the window, blackman or hamming
        :param norm_type: see class LinearTransform
        :param norm_range: see class LinearTransform
        """

        scaling = 2. * pi * f_cutoff / c

        self.window_width = float(window_width)
        self.window_type = window_type
        impulse_range = (-1.*pi *window_width/scaling, pi*window_width/scaling)
        super(self.__class__, self).__init__(scaling, impulse_range,normalization=normalization, **kwargs)
        self.label = 'Sinc filter'

    def _raw_impulse_response(self, x):
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


