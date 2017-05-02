import numpy as np
from ..core import process


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
