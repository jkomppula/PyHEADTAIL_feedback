import numpy as np
from ..core import process
from collections import deque
from scipy.constants import c

def track_beam(beam, trackers, n_turns, Q_x, Q_y=None):
    angle_x = Q_x * 2. * np.pi
    if Q_y is not None:
        angle_y = Q_y * 2. * np.pi
    else:
        angle_y = None
    
    for i in xrange(n_turns):
        #print 'Turn: ' + str(i)
        for tracker in trackers:
            tracker.operate(beam)
        
        beam.rotate(angle_x, 'x')
        if angle_y is not None:    
            beam.rotate(angle_y, 'y')

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
            print 'No isgnal!!!'
        

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

            self._beam_map = beam.charge_map

            for i in xrange(self._n_turns):
                z_values = normalized_z + float(i)*turn_length

                temp_impulse = np.interp(z_values, self._t, self._x)
                if i == 0:
                    temp_impulse[0] = 0.
                temp_impulse = np.append(np.zeros(len(temp_impulse)),temp_impulse)
                self._kick_impulses.append(temp_impulse)
                self._previous_kicks.append(np.zeros(len(normalized_z)))


        raw_source = beam.x*beam.intensity_distribution
        convolve_source = np.concatenate((raw_source,raw_source))

        for i, impulse in enumerate(self._kick_impulses):
            raw_kick=np.convolve(convolve_source,impulse, mode='full')
            i_from = len(impulse)
            i_to = len(impulse)+len(raw_source)

            if i < (self._n_turns-1):
                self._previous_kicks[i+1] += raw_kick[i_from:i_to]
            else:
                self._previous_kicks.append(raw_kick[i_from:i_to])

#        print 'self._wake_factor(beam): ' + str(self._wake_factor(beam))
#        print 'kick: '
#        print self._wake_factor(beam)*self._previous_kicks[0][self._beam_map]
#        print 'kick_done'
#        print self._previous_kicks[0][self._beam_map]
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