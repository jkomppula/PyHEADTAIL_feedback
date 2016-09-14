import numpy as np
import processors

# TODO: RandomizedParameterChange

from abc import ABCMeta, abstractmethod

class MetaProcessor(object):
    """ Basic class for derived processors
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        self.required_variables = None

    @abstractmethod
    def process(self,signal,slice_set, *args):
        pass


class RandomizedInput(MetaProcessor):
    def __init__(self,input_range, n_points, processor, input_parameters):
        super(self.__class__, self).__init__()

        self.n_points = n_points

        parameter_range = np.linspace(input_range[0],input_range[1],n_points)

        # Find the parameter from the list
        variable_index = None

        for i, val in input_parameters:
            if val == 'parameter':
                variable_index = i
                break

        self.processors = []

        for value in parameter_range:
            input_parameters[variable_index] = value
            self.processors.append(processor(*input_parameters))

        self.required_variables = self.processors[0].required_variables

    def process(self,signal,slice_set, *args):
        return self.processors[np.random.randint(0,self.n_points)].process(signal, slice_set)

class Jitter(RandomizedInput):
    def __init__(self,jitter_range, n_points=10):
        if len(jitter_range) == 1:
            input_range = (-1.*jitter_range,jitter_range)
        else:
            input_range = jitter_range

        super(self.__class__, self).__init__(input_range,n_points,processors.Delay,['parameter'])

class Bypass(MetaProcessor):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.required_variables = []

    def process(self,signal, *args):
        return signal


class  PickUpProcessor(object):
    """ A signal processor, which models a realistic two plates pickup system, which has a finite noise level and
        bandwidth. The model assumes that signals from the plates vary to opposite polarities from the reference signal
        level. The signals of both plates pass separately ChargeWeighter, NoiseGenerator and LowpassFilter in order to
        simulate realistic levels of signal, noise and frequency response. The output signal is calculated from
        the ratio of a difference and sum signals of the plates. Signals below a given threshold level is set to zero
        in order to avoid high noise level at low input signal levels

        If the cut off frequency of the LowpassFilter is higher than 'the sampling rate', a signal passes this model
        without changes. In other cases, a step response is faster than by using only a LowpassFilter but still finite.
    """

    def __init__(self,RMS_noise,f_cutoff, threshold, reference = 1e-3):
        """
        :param RMS_noise: an absolute RMS noise level in the signal [m]
        :param f_cutoff: a cutoff frequency for a signal from a single plate
        :param threshold: a relative level of the one plate signal, below which the signal is set to be zero. The reference
            level is given in the parameter reference
        :param reference: a reference signal level, when beam moves in the middle of the beam tube
        """

        self.threshold = threshold
        self.reference = reference

        self.threshold_level = 2. * self.threshold * self.reference

        self.noise_generator = processors.NoiseGenerator(RMS_noise)
        self.filter = processors.LowpassFilter(f_cutoff)
        self.charge_weighter = processors.ChargeWeighter()

    def process(self,signal,slice_set):

        signal_A = (self.reference + np.array(signal))
        signal_A = self.charge_weighter.process(signal_A,slice_set)
        signal_A = self.noise_generator.process(signal_A,slice_set)
        signal_A = self.filter.process(signal_A,slice_set)

        signal_B = (self.reference - np.array(signal))
        signal_B = self.charge_weighter.process(signal_B,slice_set)
        signal_B = self.noise_generator.process(signal_B,slice_set)
        signal_B = self.filter.process(signal_B,slice_set)

        # sets signals below the threshold level to 0. Multiplier 2 to the threshold level comes the fact that
        # the signal difference is two times the original level and the threshold level refers to the original signal
        signal_diff = signal_A - signal_B
        signal_sum = signal_A + signal_B

        signal_diff[np.absolute(signal_sum) < self.threshold_level] = 0.

        # in order to avoid 0/0, also sum signals below the threshold level have been set to 1
        signal_sum[np.absolute(signal_sum) < self.threshold_level] = 1.


        return self.reference * signal_diff / signal_sum


# class Jitter(object):
#     def __init__(self,amplitude, distribution = 'uniform',n_values = 100):
#         self.amplitude = amplitude
#         self.distribution = distribution
#         self.n_values = n_values
#
#         self.delays = np.linspace(-1.*self.amplitude,1.*self.amplitude,self.n_values)
#
#         # if self.distribution == 'normal' or self.distribution is None:
#         #     self.randoms = np.random.randn(self.values)
#         # elif self.distribution == 'uniform':
#         #     self.randoms = 1./0.577263*(-1.+2.*np.random.rand(self.values))
#         #
#         # self.randoms *= self.amplitude
#         self.processors = []
#
#         for delay in self.delays:
#             self.processors.append(Delay(delay))
#
#     def process(self, signal, slice_set):
#
#         return self.processors[randint(0,(self.n_values-1))].process(signal,slice_set)