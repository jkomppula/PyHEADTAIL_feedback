import numpy as np
from scipy.constants import c, e, m_p

from ..core import Parameters, bin_mids


class SignalObject(object):
    def __init__(self, bin_edges, intensity, ref_point = None, location_x=0., location_y=0., beta_x=1., beta_y=1.,
                 x=None, xp=None, y=None, yp=None, dp=None,
                 bunch_id=None, circumference=None, h_RF=None, circular_overlapping = 0, n_segments = 1):

        self._bunch_id = bunch_id
        self._circumference = circumference
        self._h_RF = h_RF
        self._circular_overlapping = circular_overlapping
        if (ref_point is None) and (bunch_id is not None):
            self._ref_point = self.bunch_id * self._circumference/self._h_RF
        else:
            self._ref_point = ref_point

        self._location_x = location_x
        self._location_y = location_y
        self._beta_x = beta_x
        self._beta_y = beta_y

        self._bin_edges = bin_edges
        self._z = (self._bin_edges[:, 0]+self._bin_edges[:, 1])/2.*c
        self._z_bins = np.append(self._bin_edges[:, 0], np.array([self._bin_edges[-1, 1]]))*c
        self._n_slices = len(self._z)
        self._n_segments = n_segments
        self._n_bins_per_segment = len(bin_edges)/self._n_segments

        if x is not None:
            self.x = x
        else:
            self.x = np.zeros(self._n_slices)

        if xp is not None:
            self.xp = xp
        else:
            self.xp = np.zeros(self._n_slices)

        if y is not None:
            self.y = y
        else:
            self.y = np.zeros(self._n_slices)

        if yp is not None:
            self.yp = yp
        else:
            self.yp = np.zeros(self._n_slices)

        if dp is not None:
            self.dp = dp
        else:
            self.dp = np.zeros(self._n_slices)

        if isinstance(intensity, float):
            self.intensity_distribution = np.ones(self._n_slices) * intensity
            self.intensity = intensity
        else:
            self.intensity_distribution = intensity
            self.intensity = np.sum(self.intensity_distribution)

        self.total_angle_x = 0.
        self.total_angle_y = 0.

        self._output_signal = None
        self._output_parameters = None


    def set_beam_paramters(self,p0, charge=e, mass=m_p, ):
        self.charge = charge
        self.p0 = p0
        self.mass = mass
        self.gamma = np.sqrt(1 + (p0 / (mass * c))**2)
        self.beta = np.sqrt(1 - self.gamma**-2)


    def __getattr__(self,attr):
        if (attr in ['x_amp', 'y_amp' ,'z_amp','xp_amp','yp_amp','dp_amp']):
            return self.normalized_amplitude(attr.split('_')[0])
        elif (attr in ['x_fixed','y_fixed','xp_fixed','yp_fixed']):
            return self.fixed_amplitude(attr.split('_')[0])
        else:
            return object.__getattribute__(self,attr)

    @property
    def bunch_id(self):
        return self._bunch_id

    @property
    def circumference(self):
        return self._circumference

    @property
    def h_RF(self):
        return self._h_RF

    @property
    def circular(self):
        if self._circular_overlapping > 0:
            return True
        else:
            return False

    @property
    def n_slices(self):
        return self._n_slices

    @property
    def z_bins(self):
        return self._z_bins

    @property
    def z(self):
        return self._z

    @property
    def t(self):
        return self._z/c

    @property
    def real_z(self):
        return self._z + self._bunch_id * self._circumference/float(self._h_RF)

    @property
    def real_t(self):
        return self._z/c + self._bunch_id * self._circumference/float(self._h_RF)/c

    @property
    def bin_edges(self):
        return self._bin_edges

    @property
    def n_macroparticles_per_slice(self):

        return self.intensity_distribution


    @property
    def slice_sets(self):
        return [self]

    @property
    def charge_map(self):
        """ Returns a boolean NumPy array, which gets True values for bins
            with non-zero charge
        """
        return (self.intensity_distribution != 0.)

    @property
    def epsn_x(self):
        x = np.mean(np.power(self.x, 2))
        xp = np.mean(np.power(self.xp, 2))
        x_xp = np.mean(self.x*self.xp)

        return np.sqrt(x*xp - x_xp*x_xp)

    @property
    def epsn_y(self):
        y = np.mean(np.power(self.y, 2))
        yp = np.mean(np.power(self.yp, 2))
        y_yp = np.mean(self.y*self.yp)

        return np.sqrt(y*yp - y_yp*y_yp)

    @property
    def mean_abs_x(self):
        return np.mean(np.abs(self.x))

    @property
    def mean_abs_y(self):
        return np.mean(np.abs(self.y))

    @property
    def mean_x(self):
        return np.mean(self.x)

    @property
    def mean_y(self):
        return np.mean(self.y)

    @property
    def beta_x(self):
        return self._beta_x

    @property
    def beta_y(self):
        return self._beta_y


#    def __getattr__(self,attr):
#        if (attr in ['x_amp','y_amp','xp_amp','yp_amp']):
#            return self.normalized_amplitude(axis=attr.split('_', 1 )[0])
#    #             return object.__getattr__(self,'_x')
#        elif (attr in ['x_fixed','y_fixed','xp_fixed','yp_fixed']):
#            return self.fixed_amplitude(axis=attr.split('_', 1 )[0])
#        elif (attr in ['mean_x','mean_y','mean_z','mean_xp','mean_yp','mean_dp']):
#            return object.__getattribute__(self,attr.split('_', 1 )[1])
#        else:
#            return object.__getattribute__(self,attr)

    def phase_shifted_signal(self, var, phase):

        if (phase is None) or (phase == 0.):
            return getattr(self, var)
        else:
            if var == 'x':
                return np.cos(phase)*self.x + self.beta_x*np.sin(phase)*self.xp
            elif var == 'y':
                return np.cos(phase)*self.y + self.beta_y*np.sin(phase)*self.xy
            elif var == 'xp':
                return -np.sin(phase)*self.x/self.beta_x + np.cos(phase)*self.xp
            elif var == 'yp':
                return -np.sin(phase)*self.y/self.beta_y + np.cos(phase)*self.yp
            else:
                print 'Unknown variable'


    def signal(self, var = 'x', phase_shift = None):
        overlapping = self._circular_overlapping * self._n_bins_per_segment
        if self._output_signal is None:
            self._output_signal = np.zeros(self._n_slices + 2 * overlapping)

        if self._output_parameters is None:
            if self._circular_overlapping == 0:
                bin_edges = np.copy(self._bin_edges)
            else:
                offset = self._bin_edges[-1,1] - self._bin_edges[0,0]
                bin_edges = np.concatenate((self._bin_edges[-overlapping:]-offset,
                                            self._bin_edges), axis=0)
                bin_edges = np.concatenate((bin_edges,
                                            self._bin_edges[:overlapping]+offset), axis=0)

            bunch_ref_points = []
            mids = bin_mids(bin_edges)
            for i in xrange(self._n_segments + 2*self._circular_overlapping):
                i_from = i * self._n_bins_per_segment
                i_to = (i + 1) * self._n_bins_per_segment
                bunch_ref_points.append(np.mean(mids[i_from:i_to]))

            if self._ref_point is not None:
                bunch_ref_points = bunch_ref_points - np.mean(bunch_ref_points) + self._ref_point
                bin_edges = bin_edges + self._ref_point

            self._output_parameters = Parameters(2, bin_edges, self._n_segments + 2*self._circular_overlapping, self._n_bins_per_segment,
                    bunch_ref_points,location=self._location_x, beta=self._beta_x)


        raw_signal = np.copy(self.phase_shifted_signal(var,phase_shift))

        np.copyto(self._output_signal[overlapping:(overlapping+len(raw_signal))],raw_signal)

        if self._circular_overlapping > 0 :
            np.copyto(self._output_signal[:overlapping],
                      raw_signal[-1*overlapping:])
            np.copyto(self._output_signal[-overlapping:],
                      raw_signal[:overlapping])
        return self._output_parameters, np.copy(self._output_signal)

    def correction(self,signal, beam_map=None, var='x'):
        overlapping = self._circular_overlapping * self._n_bins_per_segment

        proper_signal = signal[overlapping:(overlapping+self._n_slices)]

        if beam_map is None:
#            beam_map = np.ones(len(self._z), dtype=bool)
            beam_map = self.charge_map

        if var == 'x':
            self.x[beam_map] = self.x[beam_map] - proper_signal[beam_map]
        elif var == 'xp':
            self.xp[beam_map] = self.xp[beam_map] - proper_signal[beam_map]
        elif var == 'y':
            self.y[beam_map] = self.y[beam_map] - proper_signal[beam_map]
        elif var == 'yp':
            self.yp[beam_map] = self.yp[beam_map] - proper_signal[beam_map]
        else:
            raise ValueError('Unknown axis')

    def rotate(self, angle, axis='x'):
        s = np.sin(angle)
        c = np.cos(angle)
#        print 'self._beta_x: ' + str(self._beta_x)

        if (axis == 'x') or (axis == 'xp'):
            self.total_angle_x += angle
            new_x = c * self.x + self._beta_x * s * self.xp
            new_xp = (-1. / self._beta_x) * s * self.x + c * self.xp
            np.copyto(self.x, new_x)
            np.copyto(self.xp, new_xp)
        elif (axis == 'y') or (axis == 'yp'):
            self.total_angle_y += angle
            new_y = c * self.y + self._beta_y * s * self.yp
            new_yp = (-1. / self._beta_y) * s * self.y + c * self.yp
            np.copyto(self.y, new_y)
            np.copyto(self.yp, new_yp)
        else:
            raise ValueError('Unknown axis')

    def normalized_amplitude(self, axis='x'):
        if axis == 'x':
            return np.sqrt(self.x**2 + (self._beta_x*self.xp)**2)
        elif axis == 'y':
            return np.sqrt(self.y**2 + (self._beta_y*self.yp)**2)
        elif axis == 'xp':
            return np.sqrt(self.xp**2 + (self.x/self._beta_x)**2)
        elif axis == 'yp':
            return np.sqrt(self.yp**2 + (self.y/self._beta_y)**2)
        else:
            raise ValueError('Unknown axis')

    def fixed_amplitude(self, axis='x', offset_angle = 0.):
        if axis == 'x':
            s = np.sin(-1. * self.total_angle_x + offset_angle)
            c = np.cos(-1. * self.total_angle_x + offset_angle)
            return c * self.x + self._beta_x * s * self.xp

        elif axis == 'y':
            s = np.sin(-1. * self.total_angle_y + offset_angle)
            c = np.cos(-1. * self.total_angle_y + offset_angle)
            return c * self.y + self._beta_y * s * self.yp

        elif axis == 'xp':
            s = np.sin(-1. * self.total_angle_x + offset_angle)
            c = np.cos(-1. * self.total_angle_x + offset_angle)
            return (-1. / self._beta_x) * s * self.x + c * self.xp

        elif axis == 'yp':
            s = np.sin(-1. * self.total_angle_x + offset_angle)
            c = np.cos(-1. * self.total_angle_x + offset_angle)
            return (-1. / self._beta_y) * s * self.y + c * self.yp

        else:
            raise ValueError('Unknown axis')

    def init_noise(self,noise_level, var='x', seed = 0):
        np.random.seed(seed)
        if var == 'x':
            n_bunches = np.sum(self.charge_map)
            self.x[self.charge_map] = np.random.normal(0., noise_level, n_bunches)
            self.xp[self.charge_map] = (1. / self._beta_x)*np.random.normal(0., noise_level, n_bunches)
        else:
            raise ValueError('Unknown variable')


class Bunch(SignalObject):
    def __init__(self, length, n_slices, intensity, distribution='KV', **kwargs):
        self.length = length
        distribution = distribution
        z_bins = np.linspace(-1.*c*length/2., c*length/2., n_slices+1)
        bin_edges = np.transpose(np.array([z_bins[:-1], z_bins[1:]]))/c
        z = (bin_edges[:, 0] + bin_edges[:, 1]) / 2.

        if distribution == 'KV':
            intensity_distribution = np.ones(n_slices)*intensity/float(n_slices)
        elif distribution == 'waterbag':
            intensity_distribution = (1.-z*z/(length*length))**2
            norm_coeff = intensity/np.sum(intensity_distribution)
            intensity_distribution = intensity_distribution/norm_coeff
#        if self._distribution == 'parabolic':
#            pass
        else:
            raise ValueError('Unknown distribution!')

        super(self.__class__, self).__init__(bin_edges, intensity_distribution, **kwargs)


class Beam(object):
    def __init__(self, filling_scheme, circumference, h_RF, bunch_length, intensity, n_slices, **kwargs):
        self._filling_scheme = sorted(filling_scheme)

        self._n_slices_per_bunch = n_slices

        self._output_signal = None
        self._output_parameters = None

        self._bunch_list = []
        for bunch_id in self._filling_scheme:
            self._bunch_list.append(Bunch(bunch_length, n_slices, intensity,bunch_id=bunch_id,
                                          circumference=circumference, h_RF=h_RF,  **kwargs))

    def __getattr__(self,attr):
        if (attr in ['x','y','xp','yp','dp','x_amp','y_amp','z_amp','xp_amp','yp_amp','dp_amp', 'x_fixed','y_fixed','xp_fixed','yp_fixed','n_macroparticles_per_slice']):
            return self.combine_property(attr)
        elif attr == 'z':
            return self.combine_property('real_z')
        elif attr == 't':
            return self.combine_property('real_t')
        else:
            return object.__getattribute__(self,attr)

    def __setattr__(self, attr, value):
        if (attr in ['x','y','z','xp','yp','dp']):
            self.__set_values(value, attr)
        else:
            object.__setattr__(self,attr,value)


    @property
    def n_slices_per_bunch(self):

        return self._n_slices_per_bunch

#    @property
#    def n_macroparticles_per_slice(self):
#        return self._charge_distribution

    @property
    def slice_sets(self):
        return self._bunch_list

    def signal(self, var = 'x'):
        if self._output_parameters is None:
            bin_edges = None
            beta = None
            location = None
            segment_ref_points = []
            n_bins_per_segment = self._n_slices_per_bunch
            n_segments = len(self._bunch_list)

        if self._output_signal is None:
            self._output_signal = np.zeros(self._n_slices_per_bunch * len(self._bunch_list))

        for i, bunch in enumerate(self._bunch_list):
            parameters, signal = bunch.signal(var)

            i_from = i * self._n_slices_per_bunch
            i_to = (i + 1) * self._n_slices_per_bunch

            np.copyto(self._output_signal[i_from:i_to], signal)

            if self._output_parameters is None:
                if beta is None:
                    beta = parameters['beta']

                if location is None:
                    location = parameters['location']

                if bin_edges is None:
                    bin_edges = np.copy(parameters['bin_edges'])
                else:
#                    print 'parameters["bin_edges"]:'
#                    print parameters['bin_edges']


                    bin_edges = np.concatenate((bin_edges,parameters['bin_edges']),axis=0)

                segment_ref_points.append(parameters['segment_ref_points'][0])


        if self._output_parameters is None:
#            print 'Final edges:'
#            print bin_edges
            self._output_parameters = Parameters(signal_class=1, bin_edges=bin_edges,
                                                 n_segments=n_segments,
                                                 n_bins_per_segment=n_bins_per_segment,
                                                 segment_ref_points=segment_ref_points,
                                                 location=location,
                                                 beta=beta)

        return self._output_parameters, np.copy(self._output_signal)


    def correction(self,signal, beam_map=None, var='x'):
        for i, bunch in enumerate(self._bunch_list):

            i_from = i * self._n_slices_per_bunch
            i_to = (i + 1) * self._n_slices_per_bunch

            bunch.correction(signal[i_from:i_to],beam_map, var)

    def combine_property(self, var):

        combined = None

        for i, bunch in enumerate(self._bunch_list):
            temp_values = getattr(bunch, var)

            if combined is None:
                total_length = self._n_slices_per_bunch * len(self._bunch_list)
                combined = np.zeros(total_length)

            i_from = i*self._n_slices_per_bunch
            i_to = (i + 1)*self._n_slices_per_bunch
            np.copyto(combined[i_from:i_to], temp_values)

        return np.copy(combined)

    def __set_values(self, values, axis='x'):
        for i, bunch in enumerate(self._bunch_list):
            i_from = i * self._n_slices_per_bunch
            i_to = (i + 1) * self._n_slices_per_bunch

            setattr(bunch, axis, values[i_from:i_to])

    def rotate(self, angle, axis='x'):

        for bunch in self._bunch_list:
            bunch.rotate(angle, axis=axis)


class SimpleBeam(Beam):
    def __init__(self, n_bunches, bunch_spacing, bunch_length, intensity, n_slices, **kwargs):
        self._n_bunches = n_bunches
        self._bunch_spacing = bunch_spacing

        filling_scheme = np.arange(0, n_bunches)
        h_RF = 10*n_bunches
        circumference = h_RF * bunch_spacing*c
        super(self.__class__, self).__init__(filling_scheme, circumference,
                                             h_RF, bunch_length, intensity, n_slices, **kwargs)


class CircularPointBeam(SignalObject):
    def __init__(self, filling_scheme, circumference, h_RF, intensity, circular_overlapping,
                 **kwargs):

        z_bins = np.linspace(0,circumference,h_RF+1)
        bin_edges =np.transpose(np.array([z_bins[:-1],
                                           z_bins[1:]]))/c
        if isinstance(intensity, float):
            intensities = np.zeros(h_RF)
            for bunch_id in filling_scheme:
                intensities[bunch_id] = intensity
        elif len(filling_scheme) == len(intensity):
            for bunch_id, bunch_intensity in zip(filling_scheme,intensity):
                intensities[bunch_id] = bunch_intensity
        else:
            raise ValueError('Unknown value for intensity!')

        self._beam_map = (intensities != 0.)

        super(CircularPointBeam, self).__init__(bin_edges, intensities, bunch_id=None,
             circumference=circumference, h_RF=h_RF, circular_overlapping=circular_overlapping,
             **kwargs)

    def correction(self,signal, var='x'):
        super(CircularPointBeam, self).correction(signal, self._beam_map, var)


class SimpleCircularPointBeam(CircularPointBeam):
    def __init__(self, n_bunches, bunch_spacing, intensity, circular_overlapping, **kwargs):
        filling_scheme = np.arange(0, n_bunches)
        h_RF = n_bunches
        circumference = h_RF * bunch_spacing*c
        print circumference
        print kwargs
        super(SimpleCircularPointBeam, self).__init__(filling_scheme, circumference, h_RF,
             intensity, circular_overlapping, **kwargs)


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
    bin_edges = np.transpose(np.array([z_bins[:-1], z_bins[1:]]))/c

    x = np.zeros(len(t))
    for i, val in enumerate(t):
        if val >= 0.:
            x[i] = amplitude
            break

#    return SignalObject(bin_edges, 1., x=x, ref_point = 0.)
    return SignalObject(bin_edges, 1., x=x)


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
    bin_edges = np.transpose(np.array([z_bins[:-1], z_bins[1:]]))/c

#    return SignalObject(bin_edges, 1., x=x, ref_point = 0.)
    return SignalObject(bin_edges, 1., x=x)


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
        return np.sin(2*np.pi*x)

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
