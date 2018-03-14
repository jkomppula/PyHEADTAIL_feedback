import numpy as np
from ..core import process
from collections import deque
from scipy.constants import c
from cython_hacks import cython_circular_convolution
from scipy import signal
from abc import ABCMeta, abstractmethod


def track_beam(beam, trackers, n_turns, Q_x, Q_y=None):
    """
    A function which tracks beam by passing turn by turn the beam through the given list of
    trackers. If Q_y is given, both X- and Y-planes are tracked.

    Parameters
    ----------
    beam : int
        The beam object which is tracked
    trackers : list
        A list of trackers which operate the beam turn by turn
    n_turns : int
        A maximum number of turns tracked. If some of the trackers returns done before the given
        number of turns, the simulations is stopped.
    Q_x : float
        Tune for X-plane
    Q_y : float
        Tune for Y-plane
    """

    # betatron roation angle per turn in radians
    angle_x = Q_x * 2. * np.pi
    if Q_y is not None:
        angle_y = Q_y * 2. * np.pi
    else:
        angle_y = None

    done = False
    for i in xrange(n_turns):
        
        rotation_done = False
        # passes the beam through the trackers
        for tracker in trackers:
            tracker.operate(beam)
            done += tracker.done
            rotation_done += tracker.rotation_done
            
        if rotation_done == 0:
#            print 'I am rotating!!!'
            # Rotates beam in betatron phase
            beam.rotate(angle_x, 'x')
            if angle_y is not None:
                beam.rotate(angle_y, 'y')
#        else:
#            print 'I am not rotating!!!'

        # Tracking is stopped if some of the trackers return done
        if done > 0:
            print('Mission completed in ' + str(i) + ' turns!')
            if i < (n_turns - 2):
                break

class BeamRotator(object):

    def __init__(self, fraction , Q_x, Q_y=None):
        
        self.Q_x = Q_x
        self.Q_y = Q_y
        
        self.angle_x = Q_x * 2. * np.pi / float(fraction)
        if Q_y is not None:
            self.angle_y = Q_y * 2. * np.pi / float(fraction)
        else:
            self.angle_y = None
    
        self.rotation_done = True
        
    @property
    def done(self):
        return False
    
    def operate(self, beam, **kwargs):
        beam.rotate(self.angle_x, 'x')
        if self.angle_y is not None:
            beam.rotate(self.angle_y, 'y')

class Kicker(object):
    """ A tracker, which kicks beam after a given number of turns.
    """

    def __init__(self, kick_function, kick_turns=0, kick_var='x', seed_var='z'):
        """
        Parameters
        ----------
        kick_function : function
            A python function, which calculates the kick. The input parameter for the function
            is the beam property (Numpy array) determined by the parameter seed_var and the
            function returns a Numpy array which is added to the beam property determined in the
            parameter kick_var
        kick_turns : int or list
            A turn, when the kick is applied. If a list of numbers is given, the kick is applied
            every turn given in the list.
        kick_var : str
            A beam property which is changed by the kick, e.g. 'x', 'xp', 'y', 'yp', etc
        seed_var : str
            A beam property which is used as a seed for the kick function,
            e.g. 'z', 'x', 'xp', etc
        """

        if isinstance(kick_turns, int):
            self._kick_turns = [kick_turns]
        else:
            self._kick_turns = kick_turns

        self._kick_function = kick_function
        self._kick_var = kick_var
        self._seed_var = seed_var

        self._turn_counter = 0
        
        self.rotation_done = False

    def operate(self, beam, **kwargs):
        if self._turn_counter in self._kick_turns:
            seed = getattr(beam, self._seed_var)
            prev_values = getattr(beam, self._kick_var)
            setattr(beam, self._kick_var, prev_values + self._kick_function(seed))

        self._turn_counter += 1

    @property
    def done(self):
        return False


class AbstractTracer(object):
    __metaclass__ = ABCMeta
    """ An abstract class for tracers.
    """
    def __init__(self, n_turns, variables, triggers=None):
        """
        Parameters
        ----------
         n_turns : int
            A maximum number of turns to be stored.
        variables : list
            A list of names of beam properties to be stored. Possible properties are mean_x,
            mean_abs_x, epsn_x, mean_y, mean_abs_y and/or epsn_y
        triggers : list
            A list if tuples, which are used as triggers for storing the data. The trigger can be
            a turn number or an mean property of the beam, e.g.
                ('turn', 100) :         triggers recording after 100 turns of tracking
                ('epsn_x', 1e-10) :     triggers recording when emittance of the beam is over 1e-10
                ('mean_abs_x', 1e-3) :  triggers recording when an average displament of the beam
                                        is over 1 mm
        """
        self._n_turns = n_turns

        self.variables = variables

        # Generates buffers for the data
        self.data_tables = None
        self.tracked_turns = None

        self._n_turns_tracked = 0
        self._n_turns_stored = 0

        self._triggers = triggers

        self._triggered = False
        self._done = False
        
        self.rotation_done = False

    def _init_attributes(self, beam):
        """ Creates attributes, which allow access to the data by using variable names of the
            the original beam object
        """

        for i, var in enumerate(self.variables):
            setattr(self, self.variables[i], np.array(self.data_tables[i][:, 1:], copy=False))

        self.tracked_turns = np.array(self.data_tables[0][:, 0], copy=False)

    @abstractmethod
    def _init_data_tables(self, beam):
        """ Creates data tables
        """
        pass

    @abstractmethod
    def _update_tables(self, beam):
        """ Copies data from the beam to the data_tables.
        """
        pass

    @abstractmethod
    def save_to_file(self, file_prefix):
        """ Saves data to files
        """
        pass

    def operate(self, beam, **kwargs):

        # init variables and buffers
        if self.data_tables is None:
            self._init_data_tables(beam)
            self._init_attributes(beam)

        # checks if tracer should be triggered
        if self._triggered is False:
            self._check_for_triggering(beam)

        # if storing is triggered and not done, data are stored
        if (self._triggered is True) and (self._done is False):

            self._update_tables(beam)
            self._n_turns_stored += 1

            if self._n_turns_stored >= self._n_turns:
                self._done = True

        self._n_turns_tracked += 1

    def _check_for_triggering(self, beam):
        """ Checks if any value exceed the trigger levels
        """
        if self._triggers is not None:
            for trigger in self._triggers:
                if trigger[0] == 'turn':
                    if self._n_turns_tracked > trigger[1]:
                        self._triggered = True
                else:
                    if getattr(beam, trigger[0]) > trigger[1]:
                        self._triggered = True
        else:
            self._triggered = True

    def start_now(self):
        """ Starts recording the data
        """
        self._triggered = True
        self._done = False

    def end_now(self):
        """ Ends recording the data
        """
        self._triggered = False
        self._done = True

    def reset_data(self):
        """ Clears the data from buffers.
        """
        self._triggered = False
        self._done = False

        self._n_turns_tracked = 0
        self._n_turns_stored = 0

        if self.data_tables is not None:
            for data in self.data_tables:
                data.fill(0.)

    @property
    def done(self):
        return self._done


class AvgValueTracer(AbstractTracer):
    """ A tracker, which stores average values of the beam/bunch, e.g. emittance or average
        displacement.
    """
    def __init__(self, n_turns, variables=['mean_x','mean_abs_x','epsn_x'], **kwargs):
        """
        Parameters
        ----------
         n_turns : int
            A maximum number of turns to be stored.
        variables : list
            A list of names of beam properties to be stored. Possible properties are mean_x,
            mean_abs_x, epsn_x, mean_y, mean_abs_y and/or epsn_y
        triggers : list
            A list if tuples, which are used as triggers for storing the data. The trigger can be
            a turn number or an mean property of the beam, e.g.
                ('turn', 100) :         triggers recording after 100 turns of tracking
                ('epsn_x', 1e-10) :     triggers recording when emittance of the beam is over 1e-10
                ('mean_abs_x', 1e-3) :  triggers recording when an average displament of the beam
                                        is over 1 mm
        """
        super(self.__class__, self).__init__(n_turns, variables, **kwargs)


    def _init_data_tables(self, beam):
        self.data_tables = []
        for i in xrange(len(self.variables)):
            self.data_tables.append(np.zeros((self._n_turns, 2)))

    def _update_tables(self, beam):
        for i, var in enumerate(self.variables):
            self.data_tables[i][self._n_turns_stored, 0] = self._n_turns_tracked
            self.data_tables[i][self._n_turns_stored, 1] = getattr(beam, var)

    def save_to_file(self, file_prefix):
        output_data = np.zeros((self._n_turns_tracked, len(self.variables)+1))
        
        for i, data in enumerate(self.data_tables):
            if i == 0:
                np.copyto(output_data[:,0],data[:self._n_turns_tracked,0])
            np.copyto(output_data[:,i+1],data[:self._n_turns_tracked,1])
            
        output_data.tofile(file_prefix + '_avg_data.dat')


class Tracer(AbstractTracer):
    """ A tracer which stores slice-by-slive/bunch-by-bunch values of the bunch/beam
    """
    def __init__(self, n_turns, variables, triggers, **kwargs):
        """
        Parameters
        ----------
         n_turns : int
            A maximum number of turns to be stored.
        variables : list
            A list of names of beam properties to be stored, e.g. 'x', 'xp', etc
        triggers : list
            A list if tuples, which are used as triggers for storing the data. The trigger can be
            a turn number or an mean property of the beam, e.g.
                ('turn', 100) :         triggers recording after 100 turns of tracking
                ('epsn_x', 1e-10) :     triggers recording when emittance of the beam is over 1e-10
                ('mean_abs_x', 1e-3) :  triggers recording when an average displament of the beam
                                        is over 1 mm
        """
        super(self.__class__, self).__init__(n_turns, variables, triggers, **kwargs)


    def _init_data_tables(self, beam):
        self.data_tables = []
        self.z = np.copy(beam.z)
        for i in xrange(len(self.variables)):
            self.data_tables.append(np.zeros((self._n_turns, len(beam.z)+1)))
            setattr(self, self.variables[i], np.array(self.data_tables[-1], copy=False))

    def _update_tables(self, beam):
            for i, var in enumerate(self.variables):
                self.data_tables[i][self._n_turns_stored, 0] = self._n_turns_tracked
                np.copyto(self.data_tables[i][self._n_turns_stored, 1:], getattr(beam, var))

    def save_to_file(self, file_prefix):
        output_data = np.zeros((self._n_turns*len(self.variables)+1, len(self.z)+1))
        np.copyto(output_data[0,1:],self.z)
        
        for i in range(self._n_turns):
            for j in range(len(self.variables)):
                np.copyto(output_data[i*len(self.variables)+j+1,:],self.data_tables[j][i,:])
            
        output_data.tofile(file_prefix + '_turn_by_turn.dat')


class Damper(object):
    """ A tracer which damps beam oscillations as a trasverse damper.
    """
    def __init__(self, gain, processors, pickup_variable = 'x', kick_variable = 'x'):
        """
        Parameters
        ----------
        gain : float
            Pass band gain of the damper, i.e. 2/damping_time
        processors : list
            A list of signal processors
        pickup_variable : str
            A beam property, which is readed as a pickup signal
        kick_variable : str
            A beam property, which is kicked
        """
        self.gain = gain
        self.processors = processors
        self.pickup_variable = pickup_variable
        self.kick_variable = kick_variable
        
        self.rotation_done = False

    def operate(self, beam, **kwargs):

        # generates signal from the beam
        parameters, signal = beam.signal(self.pickup_variable)

        # processes the signal
        kick_parameters_x, kick_signal_x = process(parameters, signal, self.processors,
                                                   slice_sets=beam.slice_sets, **kwargs)
        # applies the kick
        if kick_signal_x is not None:
            kick_signal_x = kick_signal_x*self.gain
            beam.correction(kick_signal_x, var=self.kick_variable)
        else:
            print 'No signal!!!'

    @property
    def done(self):
        return False

class IdealDamper(object):
    """ A tracer which damps beam oscillations as a trasverse damper.
    """
    def __init__(self, gain, kick_variable = 'x'):
        """
        Parameters
        ----------
        gain : float
            Pass band gain of the damper, i.e. 2/damping_time
        processors : list
            A list of signal processors
        pickup_variable : str
            A beam property, which is readed as a pickup signal
        kick_variable : str
            A beam property, which is kicked
        """
        self.gain = gain
        self.kick_variable = kick_variable
        
        self.rotation_done = False

    def operate(self, beam, **kwargs):

        if self.kick_variable == 'x':
            beam.x = (1.-self.gain)*beam.x
        elif self.kick_variable == 'xp':
            beam.xp = (1.-self.gain)*beam.xp
        elif self.kick_variable == 'y':
            beam.y = (1.-self.gain)*beam.y
        elif self.kick_variable == 'yp':
            beam.yp = (1.-self.gain)*beam.yp
        else:
            raise ValueError('Unknown plane')
            

    @property
    def done(self):
        return False

class DCSuppressor(object):
    def __init__(self, n_turns=1000, source='bunch'):
        self._n_turns = float(n_turns)
        self._source = source
        
        self._dc_correction_x = None
        self._dc_correction_y = None
        
        self.rotation_done = False
        
        self._beam_map = None

    @property
    def done(self):
        return False
        
    def operate(self, beam, **kwargs):
        if self._dc_correction_x is None:
            self._beam_map = beam.charge_map
            self._dc_correction_x = np.zeros(len(beam.x))
            self._dc_correction_y = np.zeros(len(beam.y))
            self._temp_ones = np.ones(len(beam.x))
        else:
            self._dc_correction_x = (1.-1./self._n_turns)*self._dc_correction_x
            self._dc_correction_y = (1.-1./self._n_turns)*self._dc_correction_y
        
        if self._source == 'bunch':
            self._dc_correction_x = self._dc_correction_x + 1./self._n_turns * beam.x
            self._dc_correction_y = self._dc_correction_y + 1./self._n_turns * beam.y
        elif self._source == 'mean':
            self._dc_correction_x = self._dc_correction_x + 1./self._n_turns * self._temp_ones * np.mean(beam.x[self._beam_map])
            self._dc_correction_y = self._dc_correction_y + 1./self._n_turns * self._temp_ones * np.mean(beam.y[self._beam_map])
        else:
            raise ValueError('Unknown correction source!')
        
        beam.x[self._beam_map] = beam.x[self._beam_map] - self._dc_correction_x[self._beam_map]
        beam.y[self._beam_map] = beam.y[self._beam_map] - self._dc_correction_y[self._beam_map]


class CircularWake(object):
    """ A tracer which applies dipole wake kicks to the beam.
    """
    def __init__(self,wake_function, n_turns_wake, method='fft', first_bin=None, **kwargs):
        """
        Parameters
        ----------
        wake_function : function
            A function which takes z [m] values of the bins as a input parameter and returns
            the wake functions values in the units of [V/C/m]
        n_turns_wake : int
            A length of the wake function in the units of accelerator turns
        method : str
            Convolution method (affects only performance):
            'numpy': circular convultion calculated by using the linear np.convolution
            'cython': pure circular convolution programmed in Cython
            'fft': pure circular convolution by using np.ifft(np.fft(source)*np.fft(wake))
            'fftconvolve': circular convultion calculated by using the linear SciPy fftconvolve
        """

        self._wake_function = wake_function

        self._n_turns = n_turns_wake

        self._z_values = None

        self._previous_kicks = deque(maxlen=n_turns_wake)

        self._method = method
        self._first_bin = first_bin
        
        self.rotation_done = False

    @property
    def done(self):
        return False

    def _wake_factor(self, beam):
        """Universal scaling factor for the strength of a wake field
        kick from PyHEADTAIL.
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
        edges = beam.bin_edges*c
        turn_length = (edges[-1,1] - edges[0,0])
        normalized_z = (beam.z - beam.z[0])
        
        circumference = beam.circumference
        if beam.Q_x is None:
            raise ValueError('Q_x must be given to the beam object')
        angle = 2.*np.pi*(beam.Q_x%1.)*normalized_z/circumference
        self.sin = -np.sin(angle)
        self.cos =np.cos(angle)

        self._beam_map = beam.charge_map

        for i in xrange(self._n_turns):

            self._previous_kicks.append(np.zeros(len(normalized_z),dtype=complex))
            z_values = normalized_z + float(i)*turn_length

            # wake function values are interpolated from the input data
            temp_impulse = self._wake_function(z_values)
#            temp_impulse = np.interp(z_values, self._t, self._x)

            # The bunch does not kick itself, because it is assumed that the beam is below the
            # TMCI threshold. It would also difficult to determine the value for the first bin,
            # because the wake function is very time sensitive at the beging, and kick might be
            # non-constant over the first bunch
            if i == 0:
                if self._first_bin is None:
                    temp_impulse[0] = 0.
                else:
                    value = self._wake_function(np.array([self._first_bin*c]))
                    temp_impulse[0] = value[0]


            self._kick_impulses.append((impulse_modificator(self.cos*temp_impulse), impulse_modificator(self.sin*temp_impulse)))

    def operate(self, beam, **kwargs):
        if not hasattr(self, '_kick_impulses'):
            self._init(beam)

        # wake source, which is charge weighted displacement. The raw data is prepared (lengthened)
        # for the convolution method in the prepare function
        source = self._prepare_source(beam.x*beam.intensity_distribution)

        # kicks for all turns are calculated
        for i, impulse_response in enumerate(self._kick_impulses):
            kick = self._convolve(source,impulse_response[0])+1j*self._convolve(source,impulse_response[1])

            # the kicks are acculumated by adding data to values from previously tracked turns.
            # The index i+1 is used bacause, the last kick appending the list pops out
            # the first value
            if i < (self._n_turns-1):
                self._previous_kicks[i+1] += kick
            else:
                self._previous_kicks.append(kick)

        # kick is applied
        beam.xp[self._beam_map] = beam.xp[self._beam_map] + self._wake_factor(beam)*np.real(self._previous_kicks[0][self._beam_map])
        beam.x[self._beam_map] = beam.x[self._beam_map] + self._wake_factor(beam)*np.imag(self._previous_kicks[0][self._beam_map])*beam.beta_x

class CircularComplexWake(object):
    """ Version which demonstrates complex fft convolution the circular convolution wakes
     
    """
    def __init__(self,wake_function, n_turns_wake, method='fft', first_bin=None, **kwargs):
        """
        Parameters
        ----------
        wake_function : function
            A function which takes z [m] values of the bins as a input parameter and returns
            the wake functions values in the units of [V/C/m]
        n_turns_wake : int
            A length of the wake function in the units of accelerator turns
        method : str
            Convolution method (affects only performance):
            'numpy': circular convultion calculated by using the linear np.convolution
            'cython': pure circular convolution programmed in Cython
            'fft': pure circular convolution by using np.ifft(np.fft(source)*np.fft(wake))
            'fftconvolve': circular convultion calculated by using the linear SciPy fftconvolve
        """

        self._wake_function = wake_function
        self._n_turns = n_turns_wake

        self._z_values = None

        self._previous_kicks = deque(maxlen=n_turns_wake)

        self._method = method
        self._first_bin = first_bin
        
        self.rotation_done = False

    @property
    def done(self):
        return False

    def _wake_factor(self, beam):
        """Universal scaling factor for the strength of a wake field
        kick from PyHEADTAIL.
        """
        wake_factor = (-(beam.charge)**2 / (beam.mass * beam.gamma * (beam.beta * c)**2))
        return wake_factor

    def _convolve_fft(self, source, impulse_response):
            raw_kick = np.fft.ifft(np.fft.fft(source) * impulse_response)
            return raw_kick

    def _init(self, beam):
        if self._method == 'fft':
            self._convolve = self._convolve_fft
            self._prepare_source = lambda source: source
            impulse_modificator = lambda impulse: np.fft.fft(impulse)
        else:
            raise ValueError('Unknown calculation method')

        self._kick_impulses = []
        edges = beam.bin_edges*c
        turn_length = (edges[-1,1] - edges[0,0])
        normalized_z = (beam.z - beam.z[0])
        
        circumference = beam.circumference
        if beam.Q_x is None:
            raise ValueError('Q_x must be given to the beam object')
        angle = 2.*np.pi*(beam.Q_x%1.)*normalized_z/circumference
        #print angle/(2.*np.pi)
        
        self.sin = -np.sin(angle)
        self.cos =np.cos(angle)
        
#        self.sin = np.cos(angle)
#        self.cos = -np.sin(angle)
        
#        self.sin = self.sin[::-1]
#        self.cos = self.cos[::-1]

        self._beam_map = beam.charge_map

        for i in xrange(self._n_turns):

            self._previous_kicks.append(np.zeros(len(normalized_z),dtype=complex))
            z_values = normalized_z + float(i)*turn_length

            # wake function values are interpolated from the input data
            temp_impulse = self._wake_function(z_values)
#            temp_impulse = np.interp(z_values, self._t, self._x)

            # The bunch does not kick itself, because it is assumed that the beam is below the
            # TMCI threshold. It would also difficult to determine the value for the first bin,
            # because the wake function is very time sensitive at the beging, and kick might be
            # non-constant over the first bunch
            if i == 0:
                if self._first_bin is None:
                    temp_impulse[0] = 0.
                else:
                    value = self._wake_function(np.array([self._first_bin*c]))
                    temp_impulse[0] = value[0]


            self._kick_impulses.append(impulse_modificator(np.exp(-1j*angle)*temp_impulse))

    def operate(self, beam, **kwargs):
        if not hasattr(self, '_kick_impulses'):
            self._init(beam)

        # wake source, which is charge weighted displacement. The raw data is prepared (lengthened)
        # for the convolution method in the prepare function
        source = beam.x*beam.intensity_distribution + 1j*np.zeros(len(beam.x))

        # kicks for all turns are calculated
        for i, impulse_respons in enumerate(self._kick_impulses):
            kick = self._convolve(source,impulse_respons)

            # the kicks are acculumated by adding data to values from previously tracked turns.
            # The index i+1 is used bacause, the last kick appending the list pops out
            # the first value
            if i < (self._n_turns-1):
                self._previous_kicks[i+1] += kick
            else:
                self._previous_kicks.append(kick)

        # kick is applied
        kick = self._previous_kicks[0]
        
        beam.xp[self._beam_map] = beam.xp[self._beam_map] + self._wake_factor(beam)*kick.real[self._beam_map]
        beam.x[self._beam_map] = beam.x[self._beam_map] + self._wake_factor(beam)*kick.imag[self._beam_map]*beam.beta_x


class LinearWake(object):
    """ A tracer which applies dipole wake kicks to the beam.
    """
    def __init__(self,wake_function, n_turns_wake, method='numpy', first_bin=None, 
                 wake_fraction=1., **kwargs):
        """
        Parameters
        ----------
        wake_function : function
            A function which takes z [m] values of the bins as a input parameter and returns
            the wake functions values in the units of [V/C/m]
        n_turns_wake : int
            A length of the wake function in the units of accelerator turns
        method : str
            Convolution method (affects only performance):
            'numpy': circular convultion calculated by using the linear np.convolution
            'cython': pure circular convolution programmed in Cython
            'fft': pure circular convolution by using np.ifft(np.fft(source)*np.fft(wake))
            'fftconvolve': circular convultion calculated by using the linear SciPy fftconvolve
        wake_fraction : float
            A fraction of a single kick impedance model applied in this object.
            This option is for fast growth rates, when wakes are applied in multiple
            locations per turn.
        """

        self._wake_function = wake_function

        self._n_turns = n_turns_wake

        self._z_values = None

        self._previous_kicks = deque(maxlen=n_turns_wake)
        
        self._wake_fraction = float(wake_fraction)

        self._method = method
        self._first_bin = first_bin
        
        self.rotation_done = False
        self.wake_buffer = None

    @property
    def done(self):
        return False

    def _init(self, beam):
        self._kick_impulses = []
        turn_length = beam.circumference
        normalized_z = (beam.z - beam.z[0])
#        print('normalized_z: ' + str(normalized_z))

        self._beam_map = beam.charge_map
        wake_z = np.array([])



        for i in xrange(self._n_turns):
            wake_z = np.concatenate((wake_z, normalized_z + float(i)*turn_length))

        self.wake_values = self._wake_function(wake_z)/self._wake_fraction
#        print('wake_z: ' + str(wake_z))           

        if self._first_bin is None:
            self.wake_values[0] = 0.
        else:
            value = self._wake_function(np.array([self._first_bin*c]))
            self.wake_values[0] = value[0]

    def _wake_factor(self, beam):
        """Universal scaling factor for the strength of a wake field
        kick from PyHEADTAIL.
        """
        wake_factor = (-(beam.charge)**2 / (beam.mass * beam.gamma * (beam.beta * c)**2))
        return wake_factor

    
    def _rotate(self, beam, x, xp):
        accQ_x = 62.31
        angle = 2.*np.pi*accQ_x
        s = np.sin(angle)
        c = np.cos(angle)
        new_x = c * x + beam.beta_x * s * xp
        new_xp = (-1. / beam.beta_x) * s * x + c * xp

        return new_x, new_xp

    def operate(self, beam, **kwargs):
        if not hasattr(self, '_kick_impulses'):
            self._init(beam)
        
        if self.wake_buffer is None:
                self.wake_buffer = np.zeros((self._n_turns+1)*len(beam.x))
        else:
            np.copyto(self.wake_buffer[:-len(beam.x)], self.wake_buffer[len(beam.x):])
        
        
        self.wake_buffer[:-1] = self.wake_buffer[:-1] + signal.fftconvolve(self.wake_values,self._wake_factor(beam)*beam.x*beam.intensity_distribution,'full')
        temp_correction = self.wake_buffer[:len(beam.x)]
        beam.xp[self._beam_map] = beam.xp[self._beam_map] + temp_correction[self._beam_map]
        
        

def WakeSourceFromFile(filename, time_column, wake_column):
    """ Wake from a wake file
    
        Parameters
        ----------
        filename : float
            Wake filename
        time_column : int
            An index to the column including time stamps for the wake data (in the units of [ns])
        wake_column : float
            An index to the column including the wake data (in the units of [V/pC/mm])
        n_turns_wake: int
            A length of the wake function in the units of accelerator turns
    """
    wakedata = np.loadtxt(filename)
    data_t = wakedata[:, time_column]
    data_x = wakedata[:, wake_column]
    convert_to_V_per_Cm = -1e15

    def wake_function(z):
        t = z/c
        return np.interp(t, data_t*1e-9, data_x*convert_to_V_per_Cm)

    return wake_function


def ResistiveWallWakeSource(b, sigma, L):
    """ Circular resistive wall wake from an analytical formula.

        Parameters
        ----------
        b : float
            A radius of the pipe [m]
        sigma : float
            An electrical conductivity of the wall [Ohm^-1 m^-1]
        L : float
            A length of the pipe [m]
        n_turns_wake: int
            A length of the wake function in the units of accelerator turns
    """
    def wake_function(z):
        if z[0] == 0.:
            z[0] = 1e-15
        Z_0 = 119.9169832 * np.pi
        return -2./(np.pi*b**3)*np.sqrt((4.*np.pi*c)/(Z_0*c*sigma))*L/np.sqrt(z)*(Z_0*c)/(4.*np.pi)

    return wake_function

def ResonatorWakeSource(R, f_r, Q):
    """ Circular resistive wall wake from an analytical formula.

        Parameters
        ----------
        R : float
            Shunt resistance
        f_r : float
            frequency
        Q : float
            Q factor
        n_turns_wake: int
            A length of the wake function in the units of accelerator turns
    """
    def wake_function(z):
        omega_r = 2.*np.pi*f_r
        omega_r_bar = omega_r*np.sqrt(1.-1./(4.*Q**2.))
        alpha = omega_r/(2.*Q)
        t = z/c
        
        return -1.*(omega_r**2*R)/(Q*omega_r_bar)*np.exp(-1.*alpha*t)*np.sin(omega_r_bar*t)

    return wake_function