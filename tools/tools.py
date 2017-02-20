import math
import numpy as np
from scipy.constants import c, pi
from scipy import interpolate
import copy
import time

# TODO:
#   * add analytical formula for excitation
#   * multi plane for Signal handling
#   * add extract signals
#   * rewrite signal class to be based on the general abstract classes

""" This file contains tools for testing signal processors, calculating signal responses for different processor chains
    and studying the control theory behind the damping.
"""


class Signal(object):
    """ A class which emulates the SliceSet object of PyHEADTAIL without using macroparticles.
    """

    def __init__(self, z_bins, z, x, xp = None, y = None, yp=None, dp=None, Qx=None, Qy=None, Qs=None):
        """

        :param z_bins: z values of the edges of the bins/slices
        :param z: mean z (middle point) values of the bins/slices
        :param x: (mean) x values for the bins/slices
        :param xp: (mean) xp values for the bins/slices
        :param y: (mean) y values for the bins/slices
        :param yp: (mean) yp values for the bins/slices
        :param dp: (mean) dp values for the bins/slices
        :param Qx: tune in x-xp plane
        :param Qy: tune in y-yp plane
        :param Qs: tune in z/s-dp plane
        """

        self.z_bins = z_bins

        self.z = z
        self.x = x
        self.y = np.zeros(len(x))
        self.xp = np.zeros(len(x))
        self.yp = np.zeros(len(x))
        self.dp = np.zeros(len(x))

        if y is not None:
            self.y = y

        if xp is not None:
            self.xp = xp

        if yp is not None:
            self.yp = yp

        if dp is not None:
            self.dp = dp

        self.total_angle_x = 0.
        self.total_angle_y = 0.
        self.total_angle_s = 0.

        self.n_macroparticles_per_slice = np.zeros(len(x))
        self.n_macroparticles_per_slice += 1.

        self.Qx = Qx
        self.Qy = Qy
        self.Qs = Qs

        self.__signal = None

    def __call__(self, *args, **kwargs):
        return self.signal

    @property
    def signal(self):
        if self.__signal is None:
            self.pick_signal()
        return np.array(self.__signal)

    @property
    def distance(self):
        return np.array(self.z)

    @property
    def time(self):
        return self.distance/c

    @property
    def mean_x(self):
        return self.x

    @property
    def mean_xp(self):
        return self.xp

    @property
    def mean_y(self):
        return self.y

    @property
    def mean_yp(self):
        return self.yp

    @property
    def mean_z(self):
        return self.z

    @property
    def mean_dp(self):
        return self.dp

    @property
    def epsn_x(self):
        return self.x * self.x + self.xp * self.xp

    @property
    def epsn_y(self):
        return self.y * self.y + self.yp * self.yp

    @property
    def epsn_z(self):
        return self.z * self.z + self.dp * self.dp

    def set_charge_distribution(self,type, midpoint, width, max_value, parameters, threshold = 1e-3):

        """ Sets charge distribution for the Signal

        :param type: type of the distribution function. Possible options are:
            *   'fermi-dirac': Fermi-Diract distribution function. Requires one extra parameter, which discribes
                the slope on the edge of the distribution
            *   'normal': Normal (Gaussian) distribution. The parameter width is determined as 2 sigma value of
                the distribution

             Requires one parameter for the parameters
        :param midpoint: midpoint of the distribution
        :param width: width of the distribution (for the distribution function)
        :param max_value: a maximum value for the chrage
        :param parameters: extra parameters required by some distributions
        :param threshold: a threshold value (from max_value) below which the charge is set to be zero
        :return:
        """
        if type == 'fermi-dirac':
            self.n_macroparticles_per_slice= max_value / (np.exp((np.abs(self.mean_z-midpoint) - width/2.) / parameters) + 1.)

        elif type == 'normal':
            self.n_macroparticles_per_slice = max_value * np.exp(-1. * ((self.mean_z-midpoint) * (self.mean_z-midpoint)) / (0.5*0.5*width*width))

        self.n_macroparticles_per_slice[self.n_macroparticles_per_slice < threshold * max_value] = 0

    def rotate(self, planes = 'x'):
        """ Rotates signal in """

        if 'x' in planes:
            self.x, self.xp = self.__calculate_rotation(self.x, self.xp, self.Qx, self.total_angle_x)

        if 'y' in planes:
            self.y, self.yp = self.__calculate_rotation(self.y, self.yp, self.Qy, self.total_angle_y)

        if 'z' in planes:
            self.z, self.dp = self.__calculate_rotation(self.z, self.dp, self.Qs, self.total_angle_s)

    def make_correction(self, correction, plane = 'x'):
        if plane == 'x':
            self.x -= correction

        elif plane == 'y':
            self.y -= correction

        elif plane == 'z':
            self.z -= correction

    def __calculate_rotation(self,x,xp,Q,total_angle):
        """Caculate rotations required by rotate"""

        angle = Q * 2. * pi
        total_angle += angle

        c = np.cos(angle)
        s = np.sin(angle)

        prev_x = np.copy(x)
        prev_xp = np.copy(xp)

        x = c * prev_x - s * prev_xp
        xp = s * prev_x + c * prev_xp

        return (x,xp)

    def pick_signal(self,plane = 'x'):
        self.__signal =  np.array(getattr(self,plane))

    def pass_signal(self,processors):
        if isinstance(processors,list):
            for processor in processors:
                self.__signal = processor.process(self.__signal, self)
        else:
            self.__signal = processors.process(self.__signal, self)

    def bunch(self):
        return BunchEmulator(self)

class BunchEmulator(object):
    """A class which emulate a bunch object in PyHEADTAIL. Is produced from Signal object."""

    def __init__(self,signal):
        self.signal = signal

    @property
    def x(self):
        return self.signal.x

    @property
    def y(self):
        return self.signal.x

    @property
    def z(self):
        return self.signal.x

    @property
    def xp(self):
        return self.signal.x

    @property
    def yp(self):
        return self.signal.x

    @property
    def dp(self):
        return self.signal.x

    @property
    def sigma_x(self):
        return np.sqrt(np.var(self.signal.mean_x()))

    @property
    def sigma_y(self):
        return np.sqrt(np.var(self.signal.mean_y()))

    @property
    def sigma_z(self):
        return np.sqrt(np.var(self.signal.mean_z()))

    @property
    def sigma_xp(self):
        return np.sqrt(np.var(self.signal.mean_xp()))

    @property
    def sigma_yp(self):
        return np.sqrt(np.var(self.signal.mean_yp()))

    @property
    def sigma_dp(self):
        return np.sqrt(np.var(self.signal.mean_dp()))

    @property
    def mean_x(self):
        return np.mean(self.signal.mean_x())

    @property
    def mean_xp(self):
        return np.mean(self.signal.mean_xp())

    @property
    def mean_y(self):
        return np.mean(self.signal.mean_y())

    @property
    def mean_yp(self):
        return np.mean(self.signal.mean_yp())

    @property
    def mean_z(self):
        return np.mean(self.signal.mean_z())

    @property
    def mean_dp(self):
        return np.mean(self.signal.mean_dp())

    @property
    def epsn_x(self):
        return np.mean(self.signal.epsn_x())

    @property
    def epsn_y(self):
        return np.mean(self.signal.epsn_y())

    @property
    def epsn_z(self):
        return np.mean(self.signal.epsn_z())


def binary_impulse(time_range, n_points = 100, amplitude = 1.):
    """ Creates a signal where only one point has non-zero value. Can be used for calculating pure impulse responses of
        signal processors

    :param time_range: signal length in time [s]. If only one value is given, signal is from -1*value to 1* value,
            otherwise between the values given in the list or tuple
    :param n_points: number of data points in the signal
    :param amplitude: value of the non-zero point
    :return: Signal object
    """
    if isinstance(time_range, list) or isinstance(time_range, tuple):
        t_bins = np.linspace(time_range[0], time_range[1], n_points+1)
    else:
        t_bins = np.linspace(-1.*time_range, time_range, n_points + 1)

    t = np.array([(i + j) / 2. for i, j in zip(t_bins, t_bins[1:])])

    z_bins = c * t_bins
    z = c * t

    x = np.zeros(len(t))
    for i, val in enumerate(t):
        if val >= 0.:
            x[i] = amplitude
            break

    return Signal(z_bins, z, x)


def generate_signal(signal_generator, f, amplitude, n_periods, n_per_period, n_zero_periods):

    """ Abstract function which genrates signal

    :param signal_generator: a function which generates the signal. The input time unit of the function is period [t*f]
    :param f: frequency of the signal
    :param amplitude: amplitude of the signal
    :param n_periods: number of periodes included to signal
    :param n_per_period: data points per period
    :param n_zero_periods: number of periods consisting of zero values before and after the actual signal.
    :return: Signal object
    """

    t_min = -1.*n_zero_periods[0]
    t_max = n_periods+n_zero_periods[1]
    t_bins = np.linspace(t_min,t_max,int((t_max-t_min)*n_per_period)+1)
    t = np.array([(i + j) / 2. for i, j in zip(t_bins, t_bins[1:])])

    x = np.zeros(len(t))
    signal_points = (t > 0.) * (t < n_periods)
    x[signal_points] = amplitude * signal_generator(t[signal_points])

    z_bins = c * t_bins / f
    z = c * t / f

    return Signal(z_bins, z, x)


def square_signal(f, amplitude, type, duty_cycle, n_periods, n_per_period, n_zero_periods):

    """ Generates square signals, which oscillates between positive and negative value. Duty cycle describes
        a fraction of time in which the signal has non-zero value (signal can be zero finite time between positive and
        negative values, if duty cycle is below 1).
    """
    def signal_generator(x):
        signal = np.zeros(len(x))
        for i, val in enumerate(x):

            if 0.< val % 1. < duty_cycle / 2.:
                signal[i] = 1.

            elif 0.5 < val % 1. < 0.5 + duty_cycle / 2.:
                if type == 'unipolar':
                    signal[i] = 1.

                elif type == 'bipolar':
                    signal[i] = -1.

                else:
                    signal[i] = 0.
        return signal

    return generate_signal(signal_generator, f, amplitude, n_periods, n_per_period, n_zero_periods)


def square_impulse(f, amplitude = 1., duty_cycle=1., n_periods=1., n_per_period = 100, n_zero_periods = 1., type = 'bipolar'):
    """ Generates an impulse, which length is only one period by default and there are one empty period
        before and after the signal by defaul. """
    n_zero_periods = (n_zero_periods, n_zero_periods)
    return square_signal(f, amplitude, type, duty_cycle, n_periods, n_per_period, n_zero_periods)


def square_step(f, amplitude = 1., duty_cycle=1., n_periods=5., n_per_period = 100, n_zero_periods = 1., type = 'bipolar'):
    """ Generates a step impulse, which length is five periods by default and there are
         one empty period empty before the signal by defaul. """
    n_zero_periods = (n_zero_periods, 0)
    return square_signal(f, amplitude, type, duty_cycle, n_periods, n_per_period, n_zero_periods)


def square_wave(f, amplitude = 1., duty_cycle=1., n_periods=10., n_per_period = 100, type = 'bipolar'):
    """ Generates a signal, which consists of 10 pediods by defaul without empty periods before nor after the signal."""
    n_zero_periods = (0, 0)
    return square_signal(f, amplitude, type, duty_cycle, n_periods, n_per_period, n_zero_periods)


def triangle_signal(f, amplitude, n_periods, n_per_period, n_zero_periods):
    """ Generates triangular signal, which oscillates between positive and negative values.
    """
    def signal_generator(x):
        signal = np.zeros(len(x))
        for i, val in enumerate(x):
            if 0.<= val % 1. < 0.25:
                signal[i] = 4. * (val % 1.)
            elif 0.25 <= val % 1. < 0.75:
                signal[i] = 2. - 4. * (val % 1.)
            elif 0.75 <= val % 1. < 1.0:
                signal[i] = -4. + 4. * (val % 1.)

        return signal

    return generate_signal(signal_generator, f, amplitude, n_periods, n_per_period, n_zero_periods)


def triangle_impulse(f, amplitude = 1., n_periods=1., n_per_period = 100, n_zero_periods = 1.):
    n_zero_periods = (n_zero_periods, n_zero_periods)
    return triangle_signal(f, amplitude, n_periods, n_per_period, n_zero_periods)


def triangle_step(f, amplitude = 1., n_periods=5., n_per_period = 100, n_zero_periods = 1.):
    n_zero_periods = (n_zero_periods, 0)
    return triangle_signal(f, amplitude, n_periods, n_per_period, n_zero_periods)


def triangle_wave(f, amplitude = 1., n_periods=10., n_per_period = 100):
    n_zero_periods = (0, 0)
    return triangle_signal(f, amplitude, n_periods, n_per_period, n_zero_periods)


def sine_signal(f, amplitude, n_periods, n_per_period, n_zero_periods):
    """ Generates sine signal, which oscillates between positive and negative values.
    """
    def signal_generator(x):
        return np.sin(2*pi*x)

    return generate_signal(signal_generator, f, amplitude, n_periods, n_per_period, n_zero_periods)


def sine_impulse(f, amplitude = 1., n_periods=1., n_per_period = 100, n_zero_periods = 1.):
    n_zero_periods = (n_zero_periods, n_zero_periods)
    return sine_signal(f, amplitude, n_periods, n_per_period, n_zero_periods)


def sine_step(f, amplitude = 1., n_periods=5., n_per_period = 100, n_zero_periods = 1.):
    n_zero_periods = (n_zero_periods, 0)
    return sine_signal(f, amplitude, n_periods, n_per_period, n_zero_periods)


def sine_wave(f, amplitude = 1., n_periods=10., n_per_period = 100):
    n_zero_periods = (0, 0)
    return sine_signal(f, amplitude, n_periods, n_per_period, n_zero_periods)


def track_signal(n_turns, signal, processors, plane = 'x'):

    """ simulates bunch in the PyHEADTAIL, i.e.


    :param n_turns: a number of turns will be tracked
    :param signal: a signal object
    :param processors: a list of processors, 'feedback'
    :param plane: the plane where rotation occurs
    :return:
    """
    for i in xrange(n_turns):
        signal.pick_signal(plane)
        signal.pass_signal(processors)
        signal.make_correction(plane)
        signal.rotate(plane)


def signal_response(signal,processors,plane = 'x'):
    """ Calculates a response of the input signal from the given processors

    :param signal:      an input signal
    :param processors:  a list of signal processors
    :param plane:       a plane (x, y or z) where the response is calculated
    :return: (t, z, impulse, response), which are numpy arrays:
            t:          time
            z:          position
            impulse:    input signal
            response:   output signal
    """

    processors_to_use = copy.deepcopy(processors)
    signal.pick_signal(plane)

    t = signal.time
    z = signal.distance
    impulse = signal.signal

    signal.pass_signal(processors_to_use)

    response = signal.signal

    return t, z, impulse, response


def frequency_response(processors, time_scale, resp_symmetry='symmetric', f_range = None, n_f_points = 13, n_min_periods = 10, n_min_per_period = 24):

    """ Calculates a frequency response for the given processors.

    :param processors:      a list of signal processors
    :param time_scale:      a characteristic time scale of the processors (e.g. determined by cut-off frequencies of filters)
    :param resp_symmetry:   a symmetry of the impulse reponses of the processors.
    :param f_range:         a frequency range where the response is calculated
    :param n_f_points:      a number of frequencies included to the frequency range (by using logspace)
    :param n_min_periods:   a minimum number of periods included to calculation
    :param n_min_per_period: a minimum number of data points in one period
    :return: (frequencies, amplitudes, phase_shifts), which are numpy arrays
    """

    def calculate_parameters(t_period,t,signal,ref_signal):

        """ Determines the amplitude and the phase shift for the signal. Phase shift is determined by calculating
            cross correlation between a time-shifted signal and the reference signal. The time shift, which gives
            the highest correlation correspond to phase(/time) shift between the signal and the reference signal

        :param t_period: time of the period in the reference signal
        :param t: time points for signals
        :param signal: signal
        :param ref_signal: reference signal
        :return: amplitude, phase_shift
        """

        phase_steps = np.linspace(-100,100,2001)
        values = []

        ref_data_points = None

        if resp_symmetry == 'symmetric':
            start_period = np.floor((np.amax(t) - np.amin(t))/(2.*t_period))
            ref_data_points = (t >= ((start_period - 1.)* t_period))*(t < ((start_period + 1.)* t_period))
        elif resp_symmetry == 'delayed':
            ref_data_points = (t >= (np.amax(t) - 3.* t_period))*(t < (np.amax(t) - 1.* t_period))
        elif resp_symmetry == 'advanced':
            ref_data_points = (t >= (np.amin(t) + 1.* t_period))*(t < (np.amin(t) + 3.* t_period))

        ref_data_time = t[ref_data_points]
        ref_data = ref_signal[ref_data_points]
        amplitude_data = signal[ref_data_points]
        for phase in phase_steps:
            tck = interpolate.splrep(t + phase * t_period / 360., response, s=0)
            cor_data = interpolate.splev(ref_data_time, tck, der=0)
            values.append(np.sum(ref_data*cor_data))

        values = np.array(values)
        phase_shift_point = values == np.max(values)
        phase_shift = phase_steps[phase_shift_point]
        phase_shift = phase_shift[0]
        amplitude = np.amax(np.abs(amplitude_data))

        return amplitude, phase_shift

    if f_range is None:
        frequencies = np.logspace(np.log10(0.01 / time_scale), np.log10(100. / time_scale), n_f_points)
    else:
        frequencies = np.logspace(np.log10(f_range[0]),np.log10(f_range[1]),n_f_points)

    amplitudes = []
    phase_shifts = []

    for f in frequencies:
        n_periods = max(n_min_periods, int(math.ceil(1.*time_scale*f)))
        n_per_period = max(n_min_per_period,4./(time_scale*f))

        processors_for_use = copy.deepcopy(processors)
        signal = sine_wave(f,1.,n_periods,n_per_period)

        timed = signal.time
        impulse = signal.signal
        print len(timed)
        signal.pass_signal(processors_for_use)
        response = signal.signal

        amplitude, phase_shift = calculate_parameters(1./f,timed,response,impulse)

        amplitudes.append(amplitude)
        phase_shifts.append(phase_shift)
        print 'f={:.2e} -> {:.2e}, {:.2f} deg'.format(f,amplitude,phase_shift)

    return frequencies, np.array(amplitudes), np.array(phase_shifts)


def impulse_response(processors, time_range, n_points=501, impulse_type = 'binary', impulse_length = None):

    """ Calculates an impulse response for the given signal processors

    :param processors: a list of signal processors
    :param time_range: time range, where the impulse response is calculated (array or tuple)
    :param n_points: total number of points in the signal
    :param impulse_type: available options are:
            'binary': impulse consists of only one non-zero value (default)
            'square': half period of sine signal (i.e. a box signal)
            'sine': half period of sine signal (positive part)
            'triangle' half period  of triangle signal (positive part)
            'bipolar_square': one period of square signal
            'bipolar_sine': one period of sine signal
            'bipolar_triangle' one period of triangle signal
    :param impulse_length: if the type of the impulse is not binary, this parameter sets the length of the impulse. If
            empty, the impulse length is 1/10 of the time range
    :return: (t, z, impulse, response), which are numpy arrays:
            t:          time
            z:          position
            impulse:    input signal
            response:   output signal
    """

    amplitude = 1.
    signal = None

    if impulse_length is None:
        impulse_length = ((time_range[1] - time_range[0]) / 10.)

    def calculate_parameters(type):
        f = n_periods = None

        if type == 'unipolar':
            f = 0.5 / impulse_length
            n_periods = 0.5
        elif type == 'bipolar':
            f = 1. / impulse_length
            n_periods = 1.0

        total_periods = ((time_range[1] - time_range[0]) * f)
        n_per_period = int(n_points / total_periods)
        n_zero_periods = (total_periods - n_periods) / 2.

        return f, amplitude, n_periods, n_per_period, n_zero_periods

    if impulse_type == 'binary':
        signal = binary_impulse(time_range, n_points, amplitude)
    elif impulse_type == 'square':
        duty_cycle = 1.
        f, amplitude, n_periods, n_per_period, n_zero_periods = calculate_parameters('unipolar')
        signal = square_impulse(f, amplitude, duty_cycle, n_periods, n_per_period, n_zero_periods)
    elif impulse_type == 'sine':
        f, amplitude, n_periods, n_per_period, n_zero_periods = calculate_parameters('unipolar')
        signal = sine_impulse(f, amplitude, n_periods, n_per_period, n_zero_periods)
    elif impulse_type == 'triangle':
        f, amplitude, n_periods, n_per_period, n_zero_periods = calculate_parameters('unipolar')
        signal = triangle_impulse(f, amplitude, n_periods, n_per_period, n_zero_periods)
    elif impulse_type == 'bipolar_square':
        duty_cycle = 1.
        f, amplitude, n_periods, n_per_period, n_zero_periods = calculate_parameters('bipolar')
        signal = square_impulse(f, amplitude, duty_cycle, n_periods, n_per_period, n_zero_periods)
    elif impulse_type == 'bipolar_sine':
        f, amplitude, n_periods, n_per_period, n_zero_periods = calculate_parameters('bipolar')
        signal = sine_impulse(f, amplitude, n_periods, n_per_period, n_zero_periods)
    elif impulse_type == 'bipolar_triangle':
        f, amplitude, n_periods, n_per_period, n_zero_periods = calculate_parameters('bipolar')
        signal = triangle_impulse(f, amplitude, n_periods, n_per_period, n_zero_periods)

    processors_to_use = copy.deepcopy(processors)

    t = signal.time
    z = signal.distance
    impulse = signal.signal
    signal.pass_signal(processors_to_use)
    response = signal.signal

    return t, z, impulse, response



