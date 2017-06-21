import numpy as np
from ..core import process
import math, copy
from abc import ABCMeta, abstractmethod
import numpy as np
from scipy import signal
from scipy.constants import c, pi
import scipy.integrate as integrate
import scipy.special as special
from scipy.interpolate import UnivariateSpline

def kick_beam(beam, kick_function, kick_var='x', seed_var='z'):
    seed = getattr(beam, seed_var)
    setattr(beam, kick_var, kick_function(seed))


def damp_beam(beam, processors, gain, n_turns, Q, tracker=None, pickup_var='x', kicker_var=None):
    if kicker_var is None:
        kicker_var = pickup_var

    angle = Q * 2. * np.pi

    if tracker is not None:
        tracker.track(beam)
    for i in xrange(n_turns):
        parameters, signal = beam.signal(pickup_var)
        parameters, signal = process(parameters, signal, processors, slice_sets=beam.slice_sets)
        corrected_values = getattr(beam, kicker_var)
        corrected_values = corrected_values - gain * signal
        setattr(beam, kicker_var, corrected_values)
        beam.rotate(angle, pickup_var)
        if tracker is not None:
            tracker.operate(beam)

class BeamKicker(object):
    def __init__(self, kick_turns, kick_function, kick_var='x', seed_var='z'):
        if isinstance(kick_turns, int):
            self._kick_turns = [kick_turns]
        else:
            self._kick_turns = kick_turns

        self._kick_function = kick_function
        self._kick_var = kick_var
        self._seed_var = seed_var

        self._turn_counter = 0


    def operate(self, beam, **kwargs):
        if self._turn_counter in self._kick_turns:
            seed = getattr(beam, self._seed_var)
            prev_values = getattr(beam, self._kick_var)
            setattr(beam, self._kick_var, prev_values + self._kick_function(seed))

        self._turn_counter += 1


#class Resonators(object):
#    def __init__(self, frequencies, decay_times ,growth_rates, phase_shifts, seed = 0.01):
#
#        if
#
#        if isinstance(kick_turns, int):
#            self._kick_turns = [kick_turns]
#        else:
#            self._kick_turns = kick_turns
#
#        self._kick_function = kick_function
#        self._kick_var = kick_var
#        self._seed_var = seed_var
#
#        self._turn_counter = 0
#
#
#    def operate(self, beam, **kwargs):
#        if self._turn_counter in self._kick_turns:
#            seed = getattr(beam, self._seed_var)
#            prev_values = getattr(beam, self._kick_var)
#            setattr(beam, self._kick_var, prev_values + self._kick_function(seed))
#
#        self._turn_counter += 1


def beam_rotator(n_turns, gain, Q, beam, processors, workers= None, pickup_var='x', kicker_var=None):
    if isinstance(workers, object) and not isinstance(workers, list) :
        workers = [workers]

    if kicker_var is None:
        kicker_var = pickup_var

    angle = Q * 2. * np.pi

    for i in xrange(n_turns):
        parameters, signal = beam.signal(pickup_var)
        parameters, signal = process(parameters, signal, processors, slice_sets=beam.slice_sets)
        corrected_values = getattr(beam, kicker_var)
        corrected_values = corrected_values - gain * signal
        setattr(beam, kicker_var, corrected_values)
        beam.rotate(angle, pickup_var)
        if workers is not None:
            for worker in workers:
                worker.operate(beam, processors=processors)


class SignalTracker(object):
    def __init__(self):
        self._turns = []
        self._turn_counter = 0


class BeamTracker(object):
    def __init__(self,properties):
        self._properties = properties


        self._turns = []
        self._turn_counter = 0

        self._locations = None

        self._available_properties = ['x', 'y', 'xp', 'yp', 'z', 'dp',
                                      'x_amp', 'y_amp', 'xp_amp', 'yp_amp',
                                      'x_fixed', 'y_fixed', 'xp_fixed', 'yp_fixed'
                                      ]

        self._trackable_properties = [i for i in self._available_properties if i in self._properties]
        self.z = None

        if len(self._trackable_properties) > 0:
            for var in self._trackable_properties:
                setattr(self, var, [])

    def operate(self, beam, **kwargs):
        self._turns.append(self._turn_counter)
        self._turn_counter += 1

        if self.z is None:
            self.z = [np.copy(beam.z)]

        if len(self._trackable_properties) > 0:
            for var in self._trackable_properties:
                new_values = getattr(beam, var)
                getattr(self, var).append(new_values)
class Damper(object):
    def __init__(self, gain, processors):
        self._gain = gain
        self._processors = processors

    def operate(self, beam, **kwargs):
        parameters, signal = beam.damper_signal()

        parameters, signal = process(parameters, signal, self._processors)
        if signal is not None:
            beam.damper_correction(self._gain * signal)

class Wake(object):
    def __init__(self,t,x, n_turns):

        convert_to_V_per_Cm = -1e15
        self._t = t*1e-9
        self._x = x*convert_to_V_per_Cm
        self._n_turns = n_turns

        self._z_values = None
        self._kick_impulses = None

        self._previous_kicks = deque(maxlen=n_turns)

        self._kick_coeff  = 1.
	self._beam_map = None
#	self._temp_raw_kick

    def _wake_factor(self,beam):
        """Universal scaling factor for the strength of a wake field
        kick.
        """
        wake_factor = (-(beam.charge)**2 / (beam.mass * beam.gamma * (beam.beta * c)**2))
	return wake_factor

    def operate(self, beam, **kwargs):

        if self._kick_impulses is None:
            self._kick_impulses = []
            turn_length = (beam.z[-1] - beam.z[0])/c
            normalized_z = (beam.z - beam.z[0])/c

            self._beam_map = beam.intensity>0.

            for i in xrange(self._n_turns):
                z_values = normalized_z + float(i)*turn_length

                temp_impulse = np.interp(z_values, self._t, self._x)
                if i == 0:
                    temp_impulse[0] = 0.
                temp_impulse = np.append(np.zeros(len(temp_impulse)),temp_impulse)
                self._kick_impulses.append(temp_impulse)
                self._previous_kicks.append(np.zeros(len(normalized_z)))


        raw_source = beam.x*beam.intensity
        convolve_source = np.concatenate((raw_source,raw_source))

        for i, impulse in enumerate(self._kick_impulses):
            raw_kick=np.convolve(convolve_source,impulse, mode='full')
            i_from = len(impulse)
            i_to = len(impulse)+len(raw_source)

            if i < (self._n_turns-1):
                self._previous_kicks[i+1] += raw_kick[i_from:i_to]
            else:
                self._previous_kicks.append(raw_kick[i_from:i_to])


        beam.xp[self._beam_map] = beam.xp[self._beam_map] + self._wake_factor(beam)*self._previous_kicks[0][self._beam_map]


class CythonWake(object):
    def __init__(self,t,x, n_turns):

        convert_to_V_per_Cm = -1e15
        self._t = t*1e-9
        self._x = x*convert_to_V_per_Cm
        self._n_turns = n_turns

        self._z_values = None
        self._kick_impulses = None

        self._previous_kicks = deque(maxlen=n_turns)

        self._kick_coeff  = 1.
	self._beam_map = None
#	self._temp_raw_kick

    def _wake_factor(self,beam):
        """Universal scaling factor for the strength of a wake field
        kick.
        """
        wake_factor = (-(beam.charge)**2 / (beam.mass * beam.gamma * (beam.beta * c)**2))
	return wake_factor

    def operate(self, beam, **kwargs):

        if self._kick_impulses is None:
            self._kick_impulses = []
            turn_length = (beam.z[-1] - beam.z[0])/c
            normalized_z = (beam.z - beam.z[0])/c

            self._beam_map = beam.intensity>0.

            for i in xrange(self._n_turns):
                z_values = normalized_z + float(i)*turn_length

                temp_impulse = np.interp(z_values, self._t, self._x)
                if i == 0:
                    temp_impulse[0] = 0.
                self._kick_impulses.append(temp_impulse)
                self._previous_kicks.append(np.zeros(len(normalized_z)))


        raw_source = beam.x*beam.intensity
        convolve_source = raw_source

        for i, impulse in enumerate(self._kick_impulses):
            raw_kick=np.array(cython_circular_convolution(convolve_source,impulse,0))

            if i < (self._n_turns-1):
                self._previous_kicks[i+1] += raw_kick
            else:
                self._previous_kicks.append(raw_kick)


        beam.xp[self._beam_map] = beam.xp[self._beam_map] + self._wake_factor(beam)*self._previous_kicks[0][self._beam_map]