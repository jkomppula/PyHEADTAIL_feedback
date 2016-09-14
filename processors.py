import itertools
import math
import copy
from collections import deque
from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.constants import c, pi
import scipy.integrate as integrate
import scipy.special as special


""" This file contains signal processors which can be used in the feedback module in PyHEADTAIL.

    A general requirement for the signal processor is that it is a class object containing a function, namely,
    process(signal, slice_set). The input parameters for the function process(signal, slice_set) are a numpy array
    'signal' and a slice_set object of PyHEADTAIL. The function must return a numpy array with equal length to
    the input array. The other requirement is that the class object contains a list variable, namely
    'required_variables', which includes required variables for slicet_objects.

    The signals processors in this file are based on four abstract classes;
        1) in LinearTransform objects the input signal is multiplied with a matrix.
        2) in Multiplication objects the input signal is multiplied with an array with equal length to the input array
        3) in Addition objects to the input signal is added an array with equal length to the input array
        4) A normal signal processor doesn't store a signal (in terms of process() calls). Processors buffering,
           registering and/or delaying signals are namely Registers. The Registers have following properties in addition
           to the normal processor:
            a) the object is iterable
            b) the object contains a function namely combine(*args), which combines two signals returned by iteration
               together

"""

class LinearTransform(object):
    __metaclass__ = ABCMeta
    """ An abstract class for signal processors which are based on linear transformation. The signal is processed by
        calculating a dot product of a transfer matrix and a signal. The transfer matrix is produced with an abstract
        method, namely response_function(*args), which returns an elements of the matrix (an effect of
        the ref_bin to the bin)
    """

    def __init__(self, norm_type=None, norm_range=None, matrix_symmetry = 'none', bin_check = False,
                 bin_middle = 'bin', recalculate_always = False):
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

        self.norm_type = norm_type
        self.norm_range = norm_range
        self.bin_check = bin_check
        self.bin_middle = bin_middle
        self.matrix_symmetry = matrix_symmetry


        self.z_bin_set = None
        self.matrix = None

        self.recalculate_matrix = True
        self.recalculate_matrix_always = recalculate_always

        self.required_variables = ['z_bins','mean_z']

    @abstractmethod
    def response_function(self, ref_bin_mid, ref_bin_from, ref_bin_to, bin_mid, bin_from, bin_to):
        # Impulse response function of the processor
        pass

    def process(self,signal,slice_set, *args):

        # check if the bin set is changed
        if self.bin_check:
            self.recalculate_matrix = self.check_bin_set(slice_set.z_bins)

        # recalculte the matrix if necessary
        if self.recalculate_matrix:
            if self.recalculate_matrix_always == False:
                self.recalculate_matrix = False
            if self.bin_middle == 'particles':
                bin_midpoints = np.array(copy.copy(slice_set.mean_z))
            elif self.bin_middle == 'bin':
                bin_midpoints = np.array([(i+j)/2. for i, j in zip(slice_set.z_bins, slice_set.z_bins[1:])])

            self.generate_matrix(slice_set.z_bins,bin_midpoints)

        # process the signal
        return np.dot(self.matrix,signal)

    def clear_matrix(self):
        self.matrix = np.array([])
        self.recalculate_matrix = True

    def check_bin_set(self,z_bin_set):

        if self.z_bin_set is None:
            self.z_bin_set = copy.copy(z_bin_set)
            return True

        else:
            changed = False
            for old, new in itertools.izip(self.z_bin_set,z_bin_set):
                if old != new:
                    changed = True
                    self.z_bin_set = copy.copy(z_bin_set)
                    break
            return changed

    def print_matrix(self):
        for row in self.matrix:
            print "[",
            for element in row:
                print "{:6.3f}".format(element),
            print "]"

    def generate_matrix(self,bin_set, bin_midpoints):

        self.matrix = np.identity(len(bin_midpoints))
        self.matrix *= np.nan

        bin_widths = []
        for i,j in zip(bin_set,bin_set[1:]):
            bin_widths.append(j-i)

        bin_widths = np.array(bin_widths)
        bin_std = np.std(bin_widths)/np.mean(bin_widths)

        if bin_std > 1e-3:
            'Dynamic slicer -> unoptimized matrix generation!'

        if self.matrix_symmetry == 'fully_diagonal' and bin_std < 1e-3:
            j = 0
            midpoint_j = bin_midpoints[0]
            for i, midpoint_i in enumerate(bin_midpoints):
                self.matrix[j][i] = self.response_function(midpoint_i, bin_set[i], bin_set[i + 1], midpoint_j,
                                                           bin_set[j]
                                                           , bin_set[j + 1])
                for val in xrange(1,len(bin_midpoints) - max(i,j)):
                    self.matrix[j + val][i + val] = self.matrix[j][i]

            i = 0
            midpoint_i = bin_midpoints[0]
            for j, midpoint_j in enumerate(bin_midpoints[1:], start=1):
                self.matrix[j][i] = self.response_function(midpoint_i, bin_set[i], bin_set[i + 1], midpoint_j,
                                                           bin_set[j]
                                                           , bin_set[j + 1])
                for val in xrange(1,len(bin_midpoints) - max(i,j)):
                    self.matrix[j + val][i + val] = self.matrix[j][i]

        else:
            counter = 0
            for i, midpoint_i in enumerate(bin_midpoints):
                for j, midpoint_j in enumerate(bin_midpoints):
                        if np.isnan(self.matrix[j][i]):
                            counter += 1
                            self.matrix[j][i] = self.response_function(midpoint_i,bin_set[i],bin_set[i+1],midpoint_j,bin_set[j]
                                                                   ,bin_set[j+1])
            print str(counter) + ' elements is calculated'


        if self.norm_type == 'bunch_average':
            self.norm_coeff = bin_set[-1] - bin_set[0]
        elif self.norm_type == 'fixed_average':
            self.norm_coeff = self.norm_range[1] - self.norm_range[0]
        elif self.norm_type == 'bunch_integral':
            center_idx = math.floor(len(bin_midpoints) / 2)
            self.norm_coeff = self.response_function(bin_midpoints[center_idx], bin_set[center_idx],
                                                     bin_set[center_idx + 1], bin_midpoints[center_idx],
                                                     bin_set[0], bin_set[-1])
        elif self.norm_type == 'fixed_integral':
            center_idx = math.floor(len(bin_midpoints) / 2)
            self.norm_coeff = self.response_function(bin_midpoints[center_idx], bin_set[center_idx],
                                                     bin_set[center_idx + 1], bin_midpoints[center_idx],
                                                     self.norm_range[0], self.norm_range[-1])
        elif self.norm_type == 'max_column':
            self.norm_coeff= np.max(np.sum(self.matrix,0))

        elif self.norm_type is None:
            self.norm_coeff = 1.

        self.matrix = self.matrix / float(self.norm_coeff)

class BypassLinearTransform(LinearTransform):
    """ A test processor for testing the abstract class of linear transform. The response function produces
        an unit matrix
    """
    def __init__(self,norm_type = None, norm_range = None):
        super(self.__class__, self).__init__(norm_type, norm_range)

    def response_function(self, ref_bin_mid, ref_bin_from, ref_bin_to, bin_mid, bin_from, bin_to):
        if ref_bin_mid == bin_mid:
            return 1
        else:
            return 0

class Averager(LinearTransform):
    """ Returns a signal, which consists an average value of the input signal. A sums of the rows in the matrix
        are normalized to be one (i.e. a sum of the input signal doesn't change).
    """

    def __init__(self,norm_type = 'max_column', norm_range = None):
        super(self.__class__, self).__init__(norm_type, norm_range)

    def response_function(self, ref_bin_mid, ref_bin_from, ref_bin_to, bin_mid, bin_from, bin_to):
        return 1

class Delay(LinearTransform):
    """ Delays signal in units of [second].
    """
    def __init__(self,delay, norm_type = None, norm_range = None,recalculate_always = False):
        self.delay = delay
        super(self.__class__, self).__init__(norm_type, norm_range, 'fully_diagonal', recalculate_always = recalculate_always)

    def response_function(self, ref_bin_mid, ref_bin_from, ref_bin_to, bin_mid, bin_from, bin_to):

        return self.CDF(bin_to, ref_bin_from, ref_bin_to) - self.CDF(bin_from, ref_bin_from, ref_bin_to)

    def CDF(self,x,ref_bin_from, ref_bin_to):
        if (x-self.delay*c) <= ref_bin_from:
            return 0.
        elif (x-self.delay*c) < ref_bin_to:
            return ((x-self.delay*c)-ref_bin_from)/float(ref_bin_to-ref_bin_from)
        else:
            return 1.

class LinearTransformFromFile(LinearTransform):
    """ Interpolates matrix columns by using inpulse response data from a file. """

    def __init__(self,filename, x_axis = 'time', norm_type = 'max_column', norm_range = None):
        self.filename = filename
        self.x_axis = x_axis
        self.data = np.loadtxt(self.filename)
        if self.x_axis == 'time':
            self.data[:, 0]=self.data[:, 0]*c

        super(self.__class__, self).__init__(norm_type, norm_range)

    def response_function(self, ref_bin_mid, ref_bin_from, ref_bin_to, bin_mid, bin_from, bin_to):
            return np.interp(bin_mid - ref_bin_mid, self.data[:, 0], self.data[:, 1])


class Filter(LinearTransform):
    __metaclass__ = ABCMeta
    """ A general class for (analog) filters. Impulse response of the filter must be determined by overwriting
        the function raw_impulse_response.

        This processor includes two additional properties.

    """

    def __init__(self, filter_type, f_cutoff, delay, f_cutoff_2nd, norm_type, norm_range):
        """
        :param filter_type: Options are:
                'lowpass'
                'highpass'
        :param f_cutoff: cut-off frequency of the filter [Hz]
        :param delay: Delay in units of seconds
        :param f_cutoff_2nd:
        :param norm_type:
        :param norm_range:
        """

        self.f_cutoff = f_cutoff
        self.delay_z = delay * c
        self.filter_type = filter_type

        self.impulse_response = self.impulse_response_generator(f_cutoff_2nd)
        super(Filter, self).__init__(norm_type, norm_range, 'fully_diagonal')


    @abstractmethod
    def raw_impulse_response(self, x):
        """ Impulse response of the filter.
        :param x: normalized time (t*2.*pi*f_c)
        :return: response at the given time
        """
        pass

    def impulse_response_generator(self,f_cutoff_2nd):
        """ A function which generates the response function from the raw impulse response. If 2nd cut-off frequency
            is given, the value of the raw impulse response is set to constant at the time scale below that.
            The integral over the response function is normalized to value 1.
        """

        if f_cutoff_2nd is not None:
            threshold_tau = (2.*pi * self.f_cutoff) / (2.*pi * f_cutoff_2nd)
            threshold_val_neg = self.raw_impulse_response(-1.*threshold_tau)
            threshold_val_pos = self.raw_impulse_response(threshold_tau)
            integral_neg, _ = integrate.quad(self.raw_impulse_response, -100., -1.*threshold_tau)
            integral_pos, _ = integrate.quad(self.raw_impulse_response, threshold_tau, 100.)

            norm_coeff = np.abs(integral_neg + integral_pos + (threshold_val_neg + threshold_val_pos) * threshold_tau)

            def transfer_function(x):
                if np.abs(x) < threshold_tau:
                    return self.raw_impulse_response(np.sign(x)*threshold_tau)
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

        scaling = 2.*pi*self.f_cutoff/c
        temp, _ = integrate.quad(self.impulse_response, scaling * (bin_from - (ref_bin_mid+self.delay_z)),
                       scaling * (bin_to - (ref_bin_mid+self.delay_z)))

        if ref_bin_mid == bin_mid:
            if self.filter_type == 'highpass':
                temp += 1.

        return temp


class Sinc(Filter):
    """ Classical first order lowpass filter (e.g. a RC filter), which impulse response can be described as exponential
        decay.
        """
    def __init__(self, f_cutoff, delay=0., f_cutoff_2nd=None, norm_type=None, norm_range=None):
        super(self.__class__, self).__init__('lowpass', f_cutoff, delay, f_cutoff_2nd, norm_type, norm_range)

    def raw_impulse_response(self, x):
        if np.abs(x/pi) > 5.:
            return 0.
        else:
            return np.sinc(x/pi)*(0.42-0.5*np.cos(2.*pi*(x/pi+5.)/(10.))+0.08*np.cos(4.*pi*(x/pi+5.)/(10.)))


class Lowpass(Filter):
    """ Classical first order lowpass filter (e.g. a RC filter), which impulse response can be described as exponential
        decay.
        """
    def __init__(self, f_cutoff, delay=0., f_cutoff_2nd=None, norm_type=None, norm_range=None):
        super(self.__class__, self).__init__('lowpass', f_cutoff, delay, f_cutoff_2nd, norm_type, norm_range)

    def raw_impulse_response(self, x):
        if x < 0:
            return 0
        else:
            return math.exp(-1. * x)

class Highpass(Filter):
    """The classical version of a highpass filter, which """
    def __init__(self, f_cutoff, delay=0., f_cutoff_2nd=None, norm_type=None, norm_range=None):
        super(self.__class__, self).__init__('highpass', f_cutoff, delay, f_cutoff_2nd, norm_type, norm_range)

    def raw_impulse_response(self, x):
        if x < 0:
            return 0
        else:
            return -1.*math.exp(-1. * x)

class PhaseLinearizedLowpass(Filter):
    def __init__(self, f_cutoff, delay=0., f_cutoff_2nd=None, norm_type=None, norm_range=None):
        super(self.__class__, self).__init__('lowpass', f_cutoff, delay, f_cutoff_2nd, norm_type, norm_range)

    def raw_impulse_response(self, x):
        if x == 0:
            return 0
        else:
            return special.k0(abs(x))

class Multiplication(object):
    __metaclass__ = ABCMeta
    """ An abstract class which multiplies the input signal by an array. The multiplier array is produced by taking
        a slice property (determined in the input parameter 'seed') and passing it through the abstract method, namely
        multiplication_function(seed).
    """
    def __init__(self, seed, normalization = None, recalculate_multiplier = False):
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

        self.seed = seed
        self.normalization = normalization
        self.recalculate_multiplier = recalculate_multiplier

        self.multiplier = None

        self.required_variables = ['z_bins']

        if self.seed not in ['bin_length','bin_midpoint','signal']:
            self.required_variables.append(self.seed)

    @abstractmethod
    def multiplication_function(self, seed):
        pass

    def process(self,signal,slice_set):

        if (self.multiplier is None) or self.recalculate_multiplier:
            self.calculate_multiplier(signal,slice_set)

        return self.multiplier*signal

    def calculate_multiplier(self,signal,slice_set):
        if self.seed == 'bin_length':
            bin_set = slice_set.z_bins
            self.multiplier = np.array([(j-i) for i, j in zip(bin_set, bin_set[1:])])
        elif self.seed == 'bin_midpoint':
            bin_set = slice_set.z_bins
            self.multiplier = np.array([(i+j)/2. for i, j in zip(bin_set, bin_set[1:])])
        elif self.seed == 'signal':
            self.multiplier = np.array(signal)
        else:
            self.multiplier = np.array(getattr(slice_set,self.seed))

        self.multiplier = self.multiplication_function(self.multiplier)

        if self.normalization == 'total_weight':
            norm_coeff = float(np.sum(self.multiplier))
        elif self.normalization == 'average_weight':
            norm_coeff = float(np.sum(self.multiplier))/float(len(self.multiplier))
        elif self.normalization == 'maximum_weight':
            norm_coeff = float(np.max(self.multiplier))
        elif self.normalization == 'minimum_weight':
            norm_coeff = float(np.min(self.multiplier))
        elif self.normalization == None:
            norm_coeff = 1.

        self.multiplier = self.multiplier / norm_coeff

class BypassMultiplication(Multiplication):
    """
    A test processor for testing the abstract class of multiplication
    """
    def __init__(self, normalization = 'maximum_weight'):
        super(self.__class__, self).__init__('signal', normalization)

    def multiplication_function(self,weight):
        return 1.


class ChargeWeighter(Multiplication):
    """ weights signal with charge (macroparticles) of slices
    """

    def __init__(self, normalization = 'maximum_weight'):
        super(self.__class__, self).__init__('n_macroparticles_per_slice', normalization)

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
        self.bunch_length = bunch_length
        self.bunch_decay_length = bunch_decay_length
        self.maximum_weight=maximum_weight
        super(self.__class__, self).__init__('bin_midpoint', 'minimum_weight')

    def multiplication_function(self,weight):
        weight = np.exp((np.absolute(weight)-self.bunch_length/2.)/float(self.bunch_decay_length))+ 1.
        weight = np.clip(weight,1.,self.maximum_weight)
        return weight


class NoiseGate(Multiplication):
    """ Passes a signal which is greater/less than the threshold level.
    """

    def __init__(self,threshold, operator = 'greater', threshold_ref = 'amplitude'):

        self.threshold = threshold
        self.operator = operator
        self.threshold_ref = threshold_ref
        super(self.__class__, self).__init__('signal', None,recalculate_multiplier = True)

    def multiplication_function(self, seed):
        multiplier = np.zeros(len(seed))

        if self.threshold_ref == 'amplitude':
            comparable = np.abs(seed)
        elif self.threshold_ref == 'absolute':
            comparable = seed

        if self.operator == 'greater':
            multiplier[comparable > self.threshold] = 1
        elif self.operator == 'less':
            multiplier[comparable < self.threshold] = 1

        return multiplier


class MultiplicationFromFile(Multiplication):
    """ Multiplies the signal with an array, which is produced by interpolation from the loaded data. Note the seed for
        the interpolation can be any of those presented in the abstract function. E.g. a spatial weight can be
        determined by using a bin midpoint as a seed, nonlinear amplification can be modelled by using signal itself
        as a seed and etc...
    """

    def __init__(self,filename, x_axis='time', seed='bin_midpoint',normalization = None, recalculate_multiplier = False):
        super(self.__class__, self).__init__(seed, normalization, recalculate_multiplier)
        self.filename = filename
        self.x_axis = x_axis
        self.data = np.loadtxt(self.filename)
        if self.x_axis == 'time':
            self.data[:, 0] = self.data[:, 0] * c

    def multiplication_function(self, seed):
        return np.interp(seed, self.data[:, 0], self.data[:, 1])


class Addition(object):
    __metaclass__ = ABCMeta
    """ An abstract class which adds an array to the input signal. The addend array is produced by taking
        a slice property (determined in the input parameter 'seed') and passing it through the abstract method, namely
        addend_function(seed).
    """

    def __init__(self, seed, normalization = None, recalculate_addend = False):
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

        self.seed = seed
        self.normalization = normalization
        self.recalculate_addend = recalculate_addend

        self.addend = None

        self.required_variables=['z_bins']

        if self.seed not in ['bin_length','bin_midpoint','signal']:
            self.required_variables.append(self.seed)

    @abstractmethod
    def addend_function(self, seed):
        pass

    def process(self,signal,slice_set):

        if (self.addend is None) or self.recalculate_addend:
            self.calculate_addend(signal,slice_set)

        return signal + self.addend

    def calculate_addend(self,signal,slice_set):
        if self.seed == 'bin_length':
            bin_set = slice_set.z_bins
            self.addend = np.array([(j-i) for i, j in zip(bin_set, bin_set[1:])])
        elif self.seed == 'bin_midpoint':
            bin_set = slice_set.z_bins
            self.addend = np.array([(i+j)/2. for i, j in zip(bin_set, bin_set[1:])])
        elif self.seed == 'signal':
            self.addend = np.array(signal)
        else:
            self.addend = np.array(getattr(slice_set,self.seed))

        self.addend = self.addend_function(self.addend)

        if self.normalization == 'total':
            norm_coeff = float(np.sum(self.addend))
        elif self.normalization == 'average':
            norm_coeff = float(np.sum(self.addend))/float(len(self.addend))
        elif self.normalization == 'maximum':
            norm_coeff = float(np.max(self.addend))
        elif self.normalization == 'minimum':
            norm_coeff = float(np.min(self.addend))
        else:
            norm_coeff = 1.

        self.addend = self.addend / norm_coeff


class BypassAddition(Addition):
    def __init__(self, normalization = 'maximum_weight'):
        super(self.__class__, self).__init__('signal', normalization)

    def addend_function(self,weight):
        return 0.


class NoiseGenerator(Addition):
    """ Adds noise to a signal. The noise level is given as RMS value of the absolute level (reference_level = 'absolute'),
        a relative RMS level to the maximum signal (reference_level = 'maximum') or a relative RMS level to local
        signal values (reference_level = 'local'). Options for the noise distribution are a Gaussian normal distribution
        (distribution = 'normal') and an uniform distribution (distribution = 'uniform')
    """

    def __init__(self,RMS_noise_level,reference_level = 'absolute', distribution = 'normal'):

        self.RMS_noise_level = RMS_noise_level
        self.reference_level = reference_level
        self.distribution = distribution

        super(self.__class__, self).__init__('signal', None, True)

    def addend_function(self,seed):

        randoms = np.zeros(len(seed))

        if self.distribution == 'normal' or self.distribution is None:
            randoms = np.random.randn(len(seed))
        elif self.distribution == 'uniform':
            randoms = 1./0.577263*(-1.+2.*np.random.rand(len(seed)))

        if self.reference_level == 'absolute':
            addend = self.RMS_noise_level*randoms
        elif self.reference_level == 'maximum':
            addend = self.RMS_noise_level*np.max(seed)*randoms
        elif self.reference_level == 'local':
            addend = seed*self.RMS_noise_level*randoms

        return addend

class AdditionFromFile(Addition):
    """ Adds an array to the signal, which is produced by interpolation from the loaded data. Note the seed for
        the interpolation can be any of those presented in the abstract function.
    """


    def __init__(self,filename, x_axis='time', seed='bin_midpoint',normalization = None, recalculate_multiplier = False):
        super(self.__class__, self).__init__(seed, normalization, recalculate_multiplier)
        self.filename = filename
        self.x_axis = x_axis
        self.data = np.loadtxt(self.filename)
        if self.x_axis == 'time':
            self.data[:, 0] = self.data[:, 0] * c

    def addend_function(self, seed):
        return np.interp(seed, self.data[:, 0], self.data[:, 1])

class Register(object):
    __metaclass__ = ABCMeta

    """ An abstract class for a signal register. A signal is stored to the register, when the function process() is
        called. The register is iterable and returns values which have been kept in register longer than
        delay requires. Normally this means that a number of returned signals corresponds to a paremeter avg_length, but
        it is less during the first turns. The values from the register can be calculated together by using a abstract
        function combine(*). It manipulates values (in terms of a phase advance) such way they can be calculated
        together in the reader position.

        When the register is a part of a signal processor chain, the function process() returns np.array() which
        is an average of register values determined by a paremeter avg_length. The exact functionality of the register
        is determined by in the abstract iterator combine(*args).

    """

    def __init__(self, n_avg, tune, delay, position, n_slices, in_processor_chain):
        """
        :param delay: a delay between storing to reading values  in turns
        :param avg_length: a number of register values are averaged
        :param phase_shift_per_turn: a betatron phase shift per turn
        :param position: a betatron position (angle) of the register from a reference point
        :param n_slices: a length of a signal, which is returned if the register is empty
        :param in_processor_chain: if True, process() returns a signal
        """
        self.delay = delay
        self.n_avg = n_avg
        self.phase_shift_per_turn = 2.*pi * tune
        self.position = position
        self.in_processor_chain = in_processor_chain


        self.max_reg_length = self.delay+self.n_avg
        self.register = deque()

        self.n_iter_left = -1

        self.reader_position = None

        if n_slices is not None:
            self.register.append(np.zeros(n_slices))

        self.required_variables = None

    def __iter__(self):
        # calculates a maximum number of iterations. If there is no enough values in the register, sets -1, which
        # indicates that next() can return zero value

        self.n_iter_left =  len(self)
        if self.n_iter_left == 0:
            self.n_iter_left = -1
        return self

    def __len__(self):
        # returns a number of signals in the register after delay
        return max((len(self.register) - self.delay), 0)

    def next(self):
        if self.n_iter_left == 0:
            raise StopIteration
        elif self.n_iter_left == -1:
            self.n_iter_left = 0
            return (np.zeros(len(self.register[0])),None,0,self.position)
        else:
            delay = -1. * (len(self.register) - self.n_iter_left) * self.phase_shift_per_turn
            self.n_iter_left -= 1
            return (self.register[self.n_iter_left],None,delay,self.position)

    def process(self,signal, *args):

        self.register.append(signal)

        if len(self.register) > self.max_reg_length:
            self.register.popleft()

        if self.in_processor_chain == True:
            temp_signal = np.zeros(len(signal))
            if len(self) > 0:
                prev = (np.zeros(len(self.register[0])),None,0,self.position)

                for value in self:
                    combined = self.combine(value,prev,None)
                    prev = value
                    temp_signal += combined[0] / float(len(self))

            return temp_signal

    @abstractmethod
    def combine(self,x1,x2,reader_position,x_to_xp = False):

        pass


class VectorSumRegister(Register):

    def __init__(self, n_avg, tune, delay = 0, position=None, n_slices=None, in_processor_chain=True):
        self.type = 'plain'
        super(self.__class__, self).__init__(n_avg, tune, delay, position, n_slices, in_processor_chain)
        self.required_variables = []

    def combine(self,x1,x2,reader_position,x_to_xp = False):
        # determines a complex number representation from two signals (e.g. from two pickups or different turns), by using
        # knowledge about phase advance between signals. After this turns the vector to the reader's phase
        # TODO: Why not x2[3]-x1[3]?
        if (x1[3] is not None) and (x1[3] != x2[3]):
            phi_x1_x2 = x1[3]-x2[3]
        else:
            phi_x1_x2 = -1. * self.phase_shift_per_turn

        s = np.sin(phi_x1_x2)
        c = np.cos(phi_x1_x2)
        re = x1[0]
        im = (c*x1[0]-x2[0])/float(s)

        # turns the vector to the reader's position
        delta_phi = x1[2]
        if reader_position is not None:
            delta_position = x1[3] - reader_position
            delta_phi += delta_position
            if delta_position > 0:
                delta_phi -= self.phase_shift_per_turn
            if x_to_xp == True:
                delta_phi -= pi/2.

        s = np.sin(delta_phi)
        c = np.cos(delta_phi)

        return np.array([c*re-s*im,s*re+c*im])


class CosineSumRegister(Register):
    """ Returns register values by multiplying the values with a cosine of the betatron phase angle from the reader.
        If there are multiple values in different phases, the sum approaches a value equal to half of the displacement
        in the reader's position.

        The function process() returns a value, which is an average of the register values (after delay determined by
        the parameter avg_length)
    """
    def __init__(self, n_avg, tune, delay = 0, position=None, n_slices=None, in_processor_chain=True):
        self.type = 'cosine'
        super(self.__class__, self).__init__(n_avg, tune, delay, position, n_slices, in_processor_chain)
        self.required_variables = []

    def combine(self,x1,x2,reader_position,x_to_xp = False):
        #print "x_to_xp: " + str(x_to_xp)
        delta_phi = x1[2]
        if reader_position is not None:
            delta_position = self.position - reader_position
            delta_phi += delta_position
            if delta_position > 0:
                delta_phi -= self.phase_shift_per_turn
            if x_to_xp == True:
                delta_phi -= pi/2.

        return np.array([2.*math.cos(delta_phi)*x1[0],None])


