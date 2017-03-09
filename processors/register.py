import math, copy
from collections import deque
from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.constants import c, pi

from ..core import SignalParameters

# TODO: program a new register:
#   - Does not modify the signal
#   - Iteration returns value from the register
#   - delay can be set
#
# TODO: phase shift algorithms
#   - takes data from a list of registers
#   - does those signals what ever wants, can be a signal source
#
# TODO: special processors which utilize a register and algorithms
#   - delay
#   - Turn by turn fir filter


class Register(object):
    """
    Stores signals to the register. The obejct is iterable, i.e. iteration
    returns the stored signals after the given delay.
    """
    def __init__(self, n_values, tune, delay=0, store_signal=False):
        """
        Parameters
        ----------
        n_values : number
          A maximum number of signals stored and returned (in addition to
          the delay)
        tune : number
          A real number value of a betatron tune
        delay : number
          A number of turns the signal kept in the register before returning it

        """

        self._n_values = n_values
        self._delay = delay
        self._phase_advance_per_turn = 2. * np.pi * tune

        self._max_reg_length = self._n_values + self._delay
        self._n_iter_left = 0
        self._signal_register = deque()
        self._parameter_register = deque()

        self.extensions = ['store', 'register']

        self._store_signal = store_signal
        self.label = 'Register'
        self.input_signal = None
        self.input_signal_parameters = None
        self.output_signal = None
        self.output_signal_parameters = None

    @property
    def parameters(self):
        if len(self._parameter_register) > 0:
            return self._parameter_register[0]
        else:
            return None

    @property
    def phase_advance_per_turn(self):
        return self._phase_advance_per_turn

    @property
    def delay(self):
        return self._delay

    @property
    def max_length(self):
        return self._n_values

    def __len__(self):
        """
        Returns a number of signals in the register after the delay.
        """
        return max((len(self._register) - self._delay), 0)

    def __iter__(self):
        """
        Calculates how many iterations are required
        """
        self._n_iter_left = len(self)

        return self

    def next(self):
        if self._n_iter_left < 1:
            raise StopIteration

        else:
            delay = -1. * (len(self._register) - self._n_iter_left) \
                            * self._phase_shift_per_turn
            self._n_iter_left -= 1

            return (self._parameter_register[self._n_iter_left],
                    self._signal_register[self._n_iter_left], delay)

    def process(self, parameters, signal, *args, **kwargs):

        if self._store_signal:
            self.input_signal = np.copy(signal)
            self.input_parameters = copy.copy(parameters)
            self.output_signal = np.copy(signal)
            self.output_parameters = copy.copy(parameters)

        self._parameter_register.append(parameters)
        self._signal_register.append(signal)

        if len(self._parameter_register) > self._max_reg_length:
            self._parameter_register.popleft()

        if len(self._signal_register) > self._max_reg_length:
            self._signal_register.popleft()

        return parameters, signal


class Combiner(object):
    __metaclass__ = ABCMeta

    def __init__(self, registers, target_location, target_beta,
                 additional_phase_advance, store_signal = False):
        """
        Parameters
        ----------
        registers : list
          A list of registers, which are a source for the signal
        target_location : number
          A target phase advance in radians of betatron motion
        additional_phase_advance : number
          Additional phase advance for the target location.
          For example, np.pi/2. for shift from displacement in the pick up to
          divergenve in the kicker
        """

        self._registers = registers
        self._target_location = target_location
        self._target_beta = target_beta
        self._additional_phase_advance = additional_phase_advance

        self._combined_parameters = None

        self.extensions = ['store', 'combiner']

        self._store_signal = store_signal
        self.label = 'Register'
        self.input_signal = None
        self.input_parameters = None
        self.output_signal = None
        self.output_parameters = None

    @abstractmethod
    def combine(self, registers, target_location, additional_phase_advance):
        pass

    def process(self, *args, **kwargs):

        signal = self.combine(self._registers, self._target_location,
                              self._additional_phase_advance)

        if self._output_parameters is None:
            self._combined_parameters = copy.copy(registers[0].parameters)
            self._combined_parameters.additional['location'] = self._target_location
            self._combined_parameters.additional['beta'] = self._target_beta

        if self._store_signal:
            self.input_signal = None
            self.input_parameters = None
            self.output_signal = np.copy(signal)
            self.output_parameters = copy.copy(self._combined_parameters)

        return parameters, signal

# TODO: add beta correction, which depends if x -> x or x -> xp
class CosineSumCombiner(Combiner):
    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        self.label = 'Cosine sum combiner'

    def combine(self, registers, target_location, additional_phase_advance):
        combined_signal = None
        n_signals = 0

        for register in registers:
            for (parameters, signal, delay) in register:
                if combined_signal is None:
                    combined_signal = np.zeros(len(signal))
                delta_position = parameters.additional.['location'] \
                                - target_location

                if delta_position > 0:
                    delta_position -= register.phase_advance_per_turn

                delta_phi = delay + delta_position - additional_phase_advance
                n_signals += 1
                combined_signal += 2. * math.cos(delta_phi) * signal

        if combined_signal is not None:
            combined_signal = combined_signal/float(n_signals)

        return combined_signal

class HilbertCombiner(Combiner):
    def __init__(self, n_taps, *args, **kwargs):
        self._n_taps = n_taps

        self._coefficients = None
        super(self.__class__, self).__init__(*args, **kwargs)
        self.label = 'Hilbert combiner'

    def combine(self, registers, target_location, additional_phase_advance):
        if self._coefficients is None:
            self._coefficients = [None]*len(registers)

        combined_signal = None

        for i, register in enumerate(registers):
            if len(register) >= len(self._coefficients):
                if self._coefficients[i] is None:
                    self._coefficients[i] = self.generate_coefficients(
                            register, target_location,
                            additional_phase_advance)

                for j, (parameters, signal, delay) in enumerate(register):
                    if combined_signal is None:
                        combined_signal = np.zeros(len(signal))
                    combined_signal += coefficients[i][j] * signal

        combined_signal = combined_signal/float(len(registers))

        return combined_signal

    def generate_coefficients(self, register, target_location, additional_phase_advance):
        parameters = register.parameters

        delta_phi = -1. * float(register.delay) \
                    * register.phase_advance_per_turn

        delta_phi -= float(self._n_taps/2) * register.phase_advance_per_turn

        delta_position = parameters.additional.['location'] - target_location
        delta_phi += delta_position
        if delta_position > 0:
            delta_phi -= self._phase_shift_per_turn

        delta_phi -= additional_phase_advance


        coefficients = np.zeros(self._n_taps)

        for i in xrange(self._n_taps):
            n = i
            n -= self._n_taps/2
            h = 0.

            if n == 0:
                h = np.cos(delta_phi)
            elif n % 2 == 1:
                h = -2. * np.sin(delta_phi) / (pi * float(n))
            coefficients[i] = h

class VectorSumCombiner(Combiner):
    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        self.label = 'Vector sum combiner'

    def combine(self, registers, target_location, additional_phase_advance):
        # determines a complex number representation from two signals (e.g. from two pickups or different turns), by using
        # knowledge about phase advance between signals. After this turns the vector to the reader's phase
        # TODO: Why not x2[3]-x1[3]?

        combined_signal = None
        n_signals = 0

        if len(registers) == 1:
            prev_parameters = None
            prev_signal = None
            prev_delay = None

            for i, (parameters, signal, delay) in enumerate(registers[0]):
                if i == 0:
                    pass
                else:
                    delta_phi = delay - prev_delay
                    re, im = self.determine_vector(prev_signal, 0, signal,
                                                   delta_phi)

                    rotation_angle = prev_delay - delta_phi/2.
                    delta_position = parameters_1.additional['location'] - target_location
                    rotation_angle += delta_position
                    if delta_position > 0:
                        rotation_angle -= register.phase_advance_per_turn

                    calculated_signal = self.rotate_vector(re, im, rotation_angle)
                    n_signals += 1
                    combined_signal = combined_signal + calculated_signal

                prev_parameters = parameters
                prev_signal = signal
                prev_delay = delay

        elif len(registers) > 1:
            prev_register = registers[0]

            for register in resgisters:
                for (parameters_1, signal_1, delay_1), (parameters_2, signal_2, delay_2) in zip(prev_register,register):
                    phi_1 = delay_1 + parameters_1.additional['location']
                    phi_2 = delay_2 + parameters_2.additional['location']

                    delta_phi = phi_1 - phi_2

                    re, im = self.determine_vector(signal_1, signal_2, delta_phi)

                    rotation_angle = delay_1 - delta_phi/2.
                    delta_position = parameters_1.additional['location'] - target_location
                    rotation_angle += delta_position
                    if delta_position > 0:
                        rotation_angle -= register.phase_advance_per_turn

                    calculated_signal = self.rotate_vector(re, im, rotation_angle)
                    n_signals += 1
                    combined_signal = combined_signal + calculated_signal

                prev_register = register
        else:
            raise ValueError('At least one register must be given.')

        if combined_signal is not None:
            combined_signal = combined_signal/float(n_signals)

        return combined_signal

    def determine_vector(signal_1, signal_2, delta_phi):

        s = np.sin(delta_phi/2.)
        c = np.cos(delta_phi/2.)

        re = 0.5 * (signal_1 + signal_2) * (c + s * s / c)
        im = -s * signal_2 + c / s * (re - c * signal_2)

        return re, im

    def rotate_vector(re, im, angle):

        s = np.sin(angle)
        c = np.cos(angle)

        return c*re-s*im


class FIRCombiner(Combiner):
    def __init__(self, coefficients, *args, **kwargs):
        self._coefficients = coefficients
        super(self.__class__, self).__init__(*args, **kwargs)
        self.label = 'FIR combiner'

    def combine(self, registers, target_location, additional_phase_advance):
        combined_signal = None
        n_signals = 0

        for register in registers:
            if len(register) >= len(self._coefficients):
                for i, (parameters, signal, delay) in enumerate(register):
                    if combined_signal is None:
                        combined_signal = np.zeros(len(signal))
                    combined_signal += coefficients[i] * signal

        return combined_signal


class FIRTurnFilter(object):
    def __init__(self, coefficients, tune, store_signal=False):
        self._coefficients = coefficients
        self._tune = tune

        self._register = Register(len(self._coefficients), self._tune)
        self._combiner = FIRCombiner(self._coefficients)

        self.extensions = ['store', 'bunch']

        self._store_signal = store_signal
        self.label = 'TurnDelay'
        self.input_signal = None
        self.input_parameters = None
        self.output_signal = None
        self.output_parameters = None

    def process(self, parameters, signal, *args, **kwargs):
        self._register.process(parameters, signal, *args, **kwargs)

        output_parameters, output_signal = self._combiner.process(parameters,
                                                                  signal,
                                                                  *args,
                                                                  **kwargs)

        if self._store_signal:
            self.input_signal = np.copy(signal)
            self.input_parameters = copy.copy(parameters)
            self.output_signal = np.copy(output_signal)
            self.output_parameters = copy.copy(output_parameters)

        return output_parameters, output_signal


class TurnDelay(object):
    def __init__(delay, tune, n_taps=2, combiner='vector_sum',
                 additional_phase_advance=0, store_signal=False):

        self._delay = delay
        self._tune = tune
        self._n_taps = n_taps
        self._combiner_type = combiner

        self._register = Register(self._n_taps, self._tune, self._delay)
        self._combiner = None

        self.extensions = ['store', 'bunch']

        self._store_signal = store_signal
        self.label = 'TurnDelay'
        self.input_signal = None
        self.input_parameters = None
        self.output_signal = None
        self.output_parameters = None

    def process(self, parameters, signal, *args, **kwargs):
        self._register.process(parameters, signal, *args, **kwargs)

        if self._combiner is None:
            self.__init_combiner(parameters)

        output_parameters, output_signal = self._combiner.process(parameters,
                                                                  signal,
                                                                  *args,
                                                                  **kwargs)

        if self._store_signal:
            self.input_signal = np.copy(signal)
            self.input_parameters = copy.copy(parameters)
            self.output_signal = np.copy(output_signal)
            self.output_parameters = copy.copy(output_parameters)

        return output_parameters, output_signal

    def __init_combiner(parameters):
        registers = [self._register]
        target_location = parameters.addiational['location']
        target_beta = parameters.addiational[beta]
        extra_phase = self._additional_phase_advance

        if isinstance(combiner_type, str):
            if self._combiner_type == 'vector_sum':
                self._combiner = VectorSumCombiner(registers, target_location,
                                                   extra_phase)
            elif self._combiner_type == 'cosine_sum':
                self._combiner = CosineSumCombiner(registers, target_location,
                                                   extra_phase)
            elif self._combiner_type == 'hilbert':
                self._combiner = HilbertCombiner(registers, target_location,
                                                 extra_phase)
            else:
                raise ValueError('Unknown combiner type')
        else:
            self._combiner = self._combiner_type(registers, target_location,
                                                 extra_phase)




#class Register(object):
#    __metaclass__ = ABCMeta
#
#    """ An abstract class for a signal register. A signal is stored to the register, when the function process() is
#        called. The register is iterable and returns values which have been kept in register longer than
#        delay requires. Normally this means that a number of returned signals corresponds to a paremeter avg_length, but
#        it is less during the first turns. The values from the register can be calculated together by using a abstract
#        function combine(*). It manipulates values (in terms of a phase advance) such way they can be calculated
#        together in the reader position.
#
#        When the register is a part of a signal processor chain, the function process() returns np.array() which
#        is an average of register values determined by a paremeter avg_length. The exact functionality of the register
#        is determined by in the abstract iterator combine(*args).
#
#    """
#
#    def __init__(self, n_avg, tune, delay, in_processor_chain,store_signal = False):
#        """
#        :param n_avg: a number of register values (in turns) have been stored after the delay
#        :param tune: a real number value of a betatron tune (e.g. 59.28 in horizontal or 64.31 in vertical direction
#                for LHC)
#        :param delay: a delay between storing to reading values  in turns
#        :param in_processor_chain: if True, process() returns a signal
#        """
#        self.signal_parameters = None
#        self.beam_parameters = None
#        self._delay = delay
#        self._n_avg = n_avg
#        self._phase_shift_per_turn = 2.*pi * tune
#        self._in_processor_chain = in_processor_chain
#        self.combination = None
#
#        self._max_reg_length = self._delay+self._n_avg
#        self._register = deque()
#
#        self._n_iter_left = -1
#
#        self._reader_position = None
#
#        # if n_bins is not None:
#        #     self._register.append(np.zeros(n_bins))
#
#
#        self.extensions = ['store', 'register']
#
#        self.label = None
#        self._store_signal = store_signal
#        self.input_signal = None
#        self.input_signal_parameters = None
#        self.output_signal = None
#        self.output_signal_parameters = None
#
#    def __iter__(self):
#        # calculates a maximum number of iterations. If there is no enough values in the register, sets -1, which
#        # indicates that next() can return zero value
#
#        self._n_iter_left =  len(self)
#        if self._n_iter_left == 0:
#            # return None
#            self._n_iter_left = -1
#        return self
#
#    def __len__(self):
#        # returns a number of signals in the register after delay
#        return max((len(self._register) - self._delay), 0)
#
#    def next(self):
#        if self._n_iter_left < 1:
#            raise StopIteration
#        else:
#            delay = -1. * (len(self._register) - self._n_iter_left) * self._phase_shift_per_turn
#            self._n_iter_left -= 1
#            return (self._register[self._n_iter_left],None,delay,self.beam_parameters.phase_advance)
#
#    def process(self,signal_parameters, signal, *args, **kwargs):
#
#        if self._store_signal:
#            self.input_signal = np.copy(signal)
#            self.input_signal_parameters = copy.copy(signal_parameters)
#
#        if self.beam_parameters is None:
#            self.signal_parameters = signal_parameters
#            self.beam_parameters = signal_parameters.beam_parameters
#
#        self._register.append(signal)
#
#        if len(self._register) > self._max_reg_length:
#            self._register.popleft()
#
#        if self._in_processor_chain == True:
#            temp_signal = np.zeros(len(signal))
#            if len(self) > 0:
#                prev = (np.zeros(len(self._register[0])),None,0,self.beam_parameters.phase_advance)
#
#                for value in self:
#                    combined = self.combine(value,prev,None)
#                    prev = value
#                    temp_signal += combined / float(len(self))
#
#            if self._store_signal:
#                self.output_signal = np.copy(temp_signal)
#                self.output_signal_parameters = copy.copy(signal_parameters)
#
#            return signal_parameters, temp_signal
#
#    @abstractmethod
#    def combine(self,x1,x2,reader_position,x_to_xp = False):
#
#        pass


#class VectorSumRegister(Register):
#
#    def __init__(self, n_avg, tune, delay = 0, in_processor_chain=True,**kwargs):
#        self.combination = 'combined'
#        super(self.__class__, self).__init__(n_avg, tune, delay, in_processor_chain,**kwargs)
#        self.label = 'Vector sum register'
#
#    def combine(self,x1,x2,reader_phase_advance,x_to_xp = False):
#        # determines a complex number representation from two signals (e.g. from two pickups or different turns), by using
#        # knowledge about phase advance between signals. After this turns the vector to the reader's phase
#        # TODO: Why not x2[3]-x1[3]?
#
#        if (x1[3] is not None) and (x1[3] != x2[3]):
#            phi_x1_x2 = x1[3]-x2[3]
#            if phi_x1_x2 < 0:
#                # print "correction"
#                phi_x1_x2 += self._phase_shift_per_turn
#        else:
#            phi_x1_x2 = -1. * self._phase_shift_per_turn
#
#        print "Delta phi: " + str(phi_x1_x2*360./(2*pi)%360.)
#
#        s = np.sin(phi_x1_x2/2.)
#        c = np.cos(phi_x1_x2/2.)
#
#        re = 0.5 * (x1[0] + x2[0]) * (c + s * s / c)
#        im = -s * x2[0] + c / s * (re - c * x2[0])
#
#        delta_phi = x1[2]-phi_x1_x2/2.
#
#        if reader_phase_advance is not None:
#            delta_position = x1[3] - reader_phase_advance
#            delta_phi += delta_position
#            if delta_position > 0:
#                delta_phi -= self._phase_shift_per_turn
#            if x_to_xp == True:
#                delta_phi -= pi/2.
#
#        s = np.sin(delta_phi)
#        c = np.cos(delta_phi)
#
#
#        return c*re-s*im

        # An old piece. It should work as well as the code above, but it has different forbidden values for phi_x1_x2
        # (where re or im variables go to infinity). Thus it is stored to here, so that it can be found easily but it
        # will be probably removed later.
        # if (x1[3] is not None) and (x1[3] != x2[3]):
        #     phi_x1_x2 = x1[3]-x2[3]
        #     if phi_x1_x2 < 0:
        #         # print "correction"
        #         phi_x1_x2 += self._phase_shift_per_turn
        # else:
        #     phi_x1_x2 = -1. * self._phase_shift_per_turn
        #
        # s = np.sin(phi_x1_x2)
        # c = np.cos(phi_x1_x2)
        # re = x1[0]
        # im = (c*x1[0]-x2[0])/float(s)
        #
        # # turns the vector to the reader's position
        # delta_phi = x1[2]
        # if reader_phase_advance is not None:
        #     delta_position = x1[3] - reader_phase_advance
        #     delta_phi += delta_position
        #     if delta_position > 0:
        #         delta_phi -= self._phase_shift_per_turn
        #     if x_to_xp == True:
        #         delta_phi -= pi/2.
        #
        # s = np.sin(delta_phi)
        # c = np.cos(delta_phi)
        #
        # # return np.array([c*re-s*im,s*re+c*im])
        #
        # return c*re-s*im


#class CosineSumRegister(Register):
#    """ Returns register values by multiplying the values with a cosine of the betatron phase angle from the reader.
#        If there are multiple values in different phases, the sum approaches a value equal to half of the displacement
#        in the reader's position.
#    """
#    def __init__(self, n_avg, tune, delay = 0, in_processor_chain=True,**kwargs):
#
#        self.combination = 'individual'
#
#        super(self.__class__, self).__init__(n_avg, tune, delay, in_processor_chain,**kwargs)
#        self.label = 'Cosine sum register'
#
#    def combine(self,x1,x2,reader_phase_advance,x_to_xp = False):
#        delta_phi = x1[2]
#        if reader_phase_advance is not None:
#            delta_position = self.beam_parameters.phase_advance - reader_phase_advance
#            delta_phi += delta_position
#            if delta_position > 0:
#                delta_phi -= self._phase_shift_per_turn
#            if x_to_xp == True:
#                delta_phi -= pi/2.
#
#        return 2.*math.cos(delta_phi)*x1[0]
#
#class FIR_Register(Register):
#    def __init__(self, n_taps, tune, delay, zero_idx, in_processor_chain,**kwargs):
#        """ A general class for the register object, which uses FIR (finite impulse response) method to calculate
#            a correct signal for kick from the register values. Because the register can be used for multiple kicker
#            (in different locations), the filter coefficients are calculated in every call with
#            the function namely coeff_generator.
#
#        :param n_taps: length of the register (and length of filter)
#        :param tune: a real number value of a betatron tune (e.g. 59.28 in horizontal or 64.31 in vertical direction
#                for LHC)
#        :param delay: a delay between storing to reading values  in turns
#        :param zero_idx: location of the zero index of the filter coeffients
#            'middle': an index of middle value in the register is 0. Values which have spend less time than that
#                    in the register have negative indexes and vice versa
#        :param in_processor_chain: if True, process() returns a signal, if False saves computing time
#        """
#        self.combination = 'individual'
#        # self.combination = 'combined'
#        self._zero_idx = zero_idx
#        self._n_taps = n_taps
#
#        super(FIR_Register, self).__init__(n_taps, tune, delay, in_processor_chain,**kwargs)
#        self.required_variables = []
#
#    def combine(self,x1,x2,reader_phase_advance,x_to_xp = False):
#        delta_phi = -1. * float(self._delay) * self._phase_shift_per_turn
#
#        if self._zero_idx == 'middle':
#            delta_phi -= float(self._n_taps/2) * self._phase_shift_per_turn
#
#        if reader_phase_advance is not None:
#            delta_position = self.beam_parameters.phase_advance - reader_phase_advance
#            delta_phi += delta_position
#            if delta_position > 0:
#                delta_phi -= self._phase_shift_per_turn
#            if x_to_xp == True:
#                delta_phi -= pi/2.
#
#        n = self._n_iter_left
#
#        if self._zero_idx == 'middle':
#            n -= self._n_taps/2
#        # print delta_phi
#        h = self.coeff_generator(n, delta_phi)
#        h *= self._n_taps
#
#        # print str(len(self)/2) + 'n: ' + str(n) + ' -> ' + str(h)  + ' (phi = ' + str(delta_phi) + ') from ' + str(self._phase_advance) + ' to ' + str(reader_phase_advance)
#
#        return h*x1[0]
#
#    def coeff_generator(self, n, delta_phi):
#        """ Calculates filter coefficients
#        :param n: index of the value
#        :param delta_phi: total phase advance to the kicker for the value which index is 0
#        :return: filter coefficient h
#        """
#        return 0.
#
#
#class HilbertPhaseShiftRegister(FIR_Register):
#    """ A register used in some damper systems at CERN. The correct signal is calculated by using FIR phase shifter,
#    which is based on the Hilbert transform. It is recommended to use odd number of taps (e.g. 7) """
#
#    def __init__(self,n_taps, tune, delay = 0, in_processor_chain=True,**kwargs):
#        super(self.__class__, self).__init__(n_taps, tune, delay, 'middle', in_processor_chain,**kwargs)
#        self.label = 'HilbertPhaseShiftRegister'
#
#    def coeff_generator(self, n, delta_phi):
#        h = 0.
#
#        if n == 0:
#            h = np.cos(delta_phi)
#        elif n % 2 == 1:
#            h = -2. * np.sin(delta_phi) / (pi * float(n))
#
#        return h