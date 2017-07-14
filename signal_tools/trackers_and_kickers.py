import numpy as np
from ..core import process, version
from collections import deque
from scipy.constants import c
from cython_hacks import cython_circular_convolution
from scipy import signal

import matplotlib.pyplot as plt


def track_beam(beam, trackers, n_turns, Q_x, Q_y=None):
    print 'Feedback version: ' + version
    angle_x = Q_x * 2. * np.pi
    if Q_y is not None:
        angle_y = Q_y * 2. * np.pi
    else:
        angle_y = None

    done = False
    for i in xrange(n_turns):
#        print 'Turn: ' + str(i)
        for tracker in trackers:
            tracker.operate(beam)
            done += tracker.done

        beam.rotate(angle_x, 'x')
        if angle_y is not None:
            beam.rotate(angle_y, 'y')
        if done > 0:
            print 'Mission completed in ' + str(i) + ' turns!'
            if i < (n_turns - 2):
                break

class Kicker(object):
    def __init__(self, kick_function, kick_turns = 0, kick_var='x', seed_var='z'):
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

    @property
    def done(self):
        return False

class Tracer(object):
    def __init__(self,n_turns,variables='x', trace_every = 1):

        self._n_turns = n_turns
        self._counter = 0
        self._trace_every = trace_every

        if isinstance(variables, basestring):
            self.variables = [variables]
        else:
            self.variables = variables

        for var in self.variables:
            setattr(self, var, None)

    def operate(self, beam, **kwargs):
        if self._counter == 0:
            n_slices = len(beam.z)
            for var in self.variables:
                setattr(self, var, np.zeros((self._n_turns,n_slices)))

        if (self._counter < self._n_turns) and (self._counter%self._trace_every == 0):
            for var in self.variables:
                np.copyto(getattr(self,var)[self._counter,:],getattr(beam,var))

        self._counter += 1

    @property
    def done(self):
        return False


class EmittanceTracer(object):
    def __init__(self, n_turns, variables=['x'], start_from = 0):
        self._n_turns = n_turns
        self._start_from = start_from
        self._end_to = self._start_from + self._n_turns

        self.variables = variables

        self.data_tables = []
        for i in xrange(len(self.variables)):
            self.data_tables.append(np.zeros((n_turns, 2)))

        self._counter = 0

    def operate(self, beam, **kwargs):

        if (self._counter >= self._start_from) and (self._counter < self._end_to):
            idx = self._counter - self._start_from
            for i, var in enumerate(self.variables):
                self.data_tables[i][idx, 0] = self._counter
                self.data_tables[i][idx, 1] = getattr(beam, 'epsn_'+var)

        self._counter += 1

    def start_now(self):
        self._start_from = self._counter

    def end_now(self):
        self._start_from = self._counter
        self._end_to = self._start_from + self._n_turns

    def save_to_file(self, file_prefix):
        for var, data in zip(self.variables, self.data_tables):
            data.tofile(file_prefix + '_' + var + '.dat')

    def reset_data(self):
        if self.data_tables is not None:
            for data in self.data_tables:
                data.fill(0.)

    @property
    def done(self):
        if (self._counter >= self._end_to):
            return True
        else:
            return False

class DataTracer(object):
    def __init__(self, n_turns, variables=['x'], start_from = 0.,
                 lim_epsn_x=None, lim_epsn_y=None):
        self._n_turns = n_turns
        self._start_from = start_from
        self._end_to = self._start_from + self._n_turns
        self.variables = variables
        self.data_tables = None
        self._counter = 0
        self.z = None

        self._lim_epsn_x = lim_epsn_x
        self._lim_epsn_y = lim_epsn_y

        self._epsn_x_start = None
        self._epsn_y_start = None

    def operate(self, beam, **kwargs):

        if self.data_tables is None:
            self.data_tables = []
            self.z = np.copy(beam.z)
            for i in xrange(len(self.variables)):
                self.data_tables.append(np.zeros((self._n_turns, len(beam.z)+1)))

        if (self._counter >= self._start_from) and (self._counter < self._end_to):
            idx = self._counter - self._start_from
            for i, var in enumerate(self.variables):
                self.data_tables[i][idx, 0] = self._counter
                np.copyto(self.data_tables[i][idx, 1:], getattr(beam, var))
        else:
            if (self._counter < self._end_to):
                if self._lim_epsn_x is not None:
                    if beam.epsn_x > self._lim_epsn_x:
                        self.start_now()
                if self._lim_epsn_y is not None:
                    if beam.epsn_y > self._lim_epsn_y:
                        self.start_now()

        self._counter += 1

    @property
    def done(self):
        if (self._counter >= self._end_to):
            return True
        else:
            return False

    def start_now(self):
        self._start_from = self._counter
        self._end_to = self._start_from + self._n_turns

    def end_now(self):
        self._end_to = self._counter

    def reset_data(self):
        if self.data_tables is not None:
            for data in self.data_tables:
                data.fill(0.)


    def save_to_file(self, file_prefix):
        self.z.tofile(file_prefix + '_z.dat')
        for var, data in zip(self.variables, self.data_tables):
            data.tofile(file_prefix + '_' + var + '.dat')


class FixedPhaseTracer(object):
    def __init__(self,phase, variables='x', n_values=None, trace_every = 1, first_trace=0):
        pass


class Damper(object):
    def __init__(self, gain, processors, pickup_variable = 'x', kick_variable = 'x'):
        self.gain = gain
        self.processors = processors
        self.pickup_variable = pickup_variable
        self.kick_variable = kick_variable

    def operate(self, beam, **kwargs):
        parameters, signal = beam.signal(self.pickup_variable)

        kick_parameters_x, kick_signal_x = process(parameters, signal, self.processors,
                                                   slice_sets=beam.slice_sets, **kwargs)
        if kick_signal_x is not None:
            kick_signal_x = kick_signal_x*self.gain
            beam.correction(kick_signal_x, var=self.kick_variable)
        else:
            print 'No signal!!!'

    @property
    def done(self):
        return False



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


class Wake(object):
    def __init__(self,t,x, n_turns, method = 'numpy'):

        convert_to_V_per_Cm = -1e15
        self._t = t*1e-9
        self._x = x*convert_to_V_per_Cm
        self._n_turns = n_turns

        self._z_values = None

        self._previous_kicks = deque(maxlen=n_turns)

        self._method = method

    @property
    def done(self):
        return False

    def _wake_factor(self, beam):
        """Universal scaling factor for the strength of a wake field
        kick.
        """
        wake_factor = (-(beam.charge)**2 / (beam.mass * beam.gamma * (beam.beta * c)**2))
        return wake_factor

    def _convolve_numpy(self, source, impulse_response):
            raw_kick = np.convolve(source,impulse_response, mode='full')
            i_from = len(impulse_response)
            i_to = len(impulse_response)+len(source)/2
            return raw_kick[i_from:i_to]

    def _convolve_cython(self, source, impulse_response):
            raw_kick = np.array(cython_circular_convolution(source, impulse_response, 0))
            return raw_kick

    def _convolve_fft(self, source, impulse_response):
            raw_kick = np.real(np.fft.ifft(np.fft.fft(source) * impulse_response))
            return raw_kick

    def _convolve_fftconcolve(self, source, impulse_response):
            raw_kick = signal.fftconvolve(source,impulse_response, mode='full')
            i_from = len(impulse_response)
            i_to = len(impulse_response)+len(source)/2
            return raw_kick[i_from:i_to]

    def _init(self, beam):
        if self._method == 'numpy':
            self._convolve = self._convolve_numpy
            self._prepare_source = lambda source: np.concatenate((source,source))
            impulse_modificator = lambda impulse: np.append(np.zeros(len(impulse)), impulse)
        elif self._method == 'cython':
            self._convolve = self._convolve_cython
            self._prepare_source = lambda source: source
            impulse_modificator = lambda impulse: impulse
        elif self._method == 'fft':
            self._convolve = self._convolve_fft
            self._prepare_source = lambda source: source
            impulse_modificator = lambda impulse: np.fft.fft(impulse)
        elif self._method == 'fftconvolve':
            self._convolve = self._convolve_fftconcolve
            self._prepare_source = lambda source: np.concatenate((source,source))
            impulse_modificator = lambda impulse: np.append(np.zeros(len(impulse)), impulse)
        else:
            raise ValueError('Unknown calculation method')

        self._kick_impulses = []
        turn_length = (beam.z[-1] - beam.z[0])/c
        normalized_z = (beam.z - beam.z[0])/c

        self._beam_map = beam.charge_map

        for i in xrange(self._n_turns):
            self._previous_kicks.append(np.zeros(len(normalized_z)))
            z_values = normalized_z + float(i)*turn_length

            temp_impulse = np.interp(z_values, self._t, self._x)
            if i == 0:
                temp_impulse[0] = 0.

            self._kick_impulses.append(impulse_modificator(temp_impulse))

    def operate(self, beam, **kwargs):
        if not hasattr(self, '_kick_impulses'):
            self._init(beam)

        source = self._prepare_source(beam.x*beam.intensity_distribution)

        for i, impulse_response in enumerate(self._kick_impulses):
            kick = self._convolve(source,impulse_response)

            if i < (self._n_turns-1):
                self._previous_kicks[i+1] += kick
            else:
                self._previous_kicks.append(kick)

#        beam.xp = beam.xp + self._wake_factor(beam)*self._previous_kicks[0]
        beam.xp[self._beam_map] = beam.xp[self._beam_map] + self._wake_factor(beam)*self._previous_kicks[0][self._beam_map]
