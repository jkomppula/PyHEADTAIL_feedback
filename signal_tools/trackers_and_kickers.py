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

        # passes the beam through the trackers
        for tracker in trackers:
            tracker.operate(beam)
            done += tracker.done

        # Rotates beam in betatron phase
        beam.rotate(angle_x, 'x')
        if angle_y is not None:
            beam.rotate(angle_y, 'y')

        # Tracking is stopped if some of the trackers return done
        if done > 0:
            print 'Mission completed in ' + str(i) + ' turns!'
            if i < (n_turns - 2):
                break


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
        for var, data in zip(self.variables, self.data_tables):
            data.tofile(file_prefix + var + '.dat')


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
        self.z.tofile(file_prefix + '_z.dat')
        for var, data in zip(self.variables, self.data_tables):
            data.tofile(file_prefix + '_' + var + '.dat')


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


class Wake(object):
    """ A tracer which applies dipole wake kicks to the beam.
    """
    def __init__(self,wake_function, n_turns_wake, method='numpy', first_bin=None, **kwargs):
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

        self._beam_map = beam.charge_map

        for i in xrange(self._n_turns):

            self._previous_kicks.append(np.zeros(len(normalized_z)))
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


            self._kick_impulses.append(impulse_modificator(temp_impulse))

    def operate(self, beam, **kwargs):
        if not hasattr(self, '_kick_impulses'):
            self._init(beam)

        # wake source, which is charge weighted displacement. The raw data is prepared (lengthened)
        # for the convolution method in the prepare function
        source = self._prepare_source(beam.x*beam.intensity_distribution)

        # kicks for all turns are calculated
        for i, impulse_response in enumerate(self._kick_impulses):
            kick = self._convolve(source,impulse_response)

            # the kicks are acculumated by adding data to values from previously tracked turns.
            # The index i+1 is used bacause, the last kick appending the list pops out
            # the first value
            if i < (self._n_turns-1):
                self._previous_kicks[i+1] += kick
            else:
                self._previous_kicks.append(kick)

        # kick is applied
        beam.xp[self._beam_map] = beam.xp[self._beam_map] + self._wake_factor(beam)*self._previous_kicks[0][self._beam_map]


class WakesFromFile(Wake):
    """ Wake from a wake file
    """
    def __init__(self, filename, time_column, wake_column, n_turns_wake, **kwargs):
        """
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

        super(self.__class__, self).__init__(wake_function, n_turns_wake, **kwargs)


class ResistiveWallWake(Wake):
    """ Circular resistive wall wake from an analytical formula.
    """
    def __init__(self, b, sigma, L, n_turns_wake, **kwargs):
        """
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

        super(self.__class__, self).__init__(wake_function, n_turns_wake, **kwargs)


