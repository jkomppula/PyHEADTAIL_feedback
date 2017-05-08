import numpy as np
from scipy.constants import c, pi
from collections import deque
from ..core import Parameters
from scipy.constants import c, e, m_p

class Beam(object):

    def __init__(self, n_bunches, bunch_spacing, intensity,p0, n_buckets_per_bunch=1, location_x=0., location_y=0.,
                 beta_x=1., beta_y=1.):
	self.charge=e
	self.p0 = p0
        self.mass = m_p
	self.gamma = np.sqrt(1 + (self.p0 / (self.mass * c))**2)
	self.beta = np.sqrt(1 - self.gamma**-2)
        self._n_bunches = n_bunches
        self._bunch_spacing = bunch_spacing

        self._n_slices = n_bunches*n_buckets_per_bunch

        if isinstance(intensity, float):
            self._intensity = np.zeros(n_bunches*n_buckets_per_bunch)
            self._intensity[::n_buckets_per_bunch] = intensity
        elif len(intensity) == n_bunches:
            self._intensity = np.zeros(n_bunches*n_buckets_per_bunch)
            self._intensity[::n_buckets_per_bunch] = intensity
        else:
            raise ValueError('Unknown value for intensity')




        self._length = n_bunches * bunch_spacing * c
        print 'self._length:' + str(self._length)
        self._beta_x = beta_x
        self._beta_y = beta_y
        self._location_x = location_x
        self._location_y = location_y

        self._z_bins = np.linspace(0., self._length, self._n_slices+1)
        self._bin_edges = np.transpose(np.array([self._z_bins[:-1], self._z_bins[1:]]))

        self.z = (self._bin_edges[:, 0]+self._bin_edges[:, 1])/2.
        self.x = np.zeros(self._n_slices)
        self.xp = np.zeros(self._n_slices)
        self.y = np.zeros(self._n_slices)
        self.yp = np.zeros(self._n_slices)
        self.dp = np.zeros(self._n_slices)
	self.temp_xp = np.zeros(self._n_slices)
	self.temp_x = np.zeros(self._n_slices)
        self.t = self.z/c

        self._total_angle_x = 0.
        self._total_angle_y = 0.


    def __getattr__(self,attr):
        if (attr in ['x_amp','y_amp','xp_amp','yp_amp']):
            return self.normalized_amplitude(axis=attr.split('_', 1 )[0])
#             return object.__getattr__(self,'_x')
        elif (attr in ['x_fixed','y_fixed','xp_fixed','yp_fixed']):
            return self.fixed_amplitude(axis=attr.split('_', 1 )[0])
        elif (attr in ['mean_x','mean_y','mean_z','mean_xp','mean_yp','mean_dp']):
            return object.__getattribute__(self,attr.split('_', 1 )[1])
        else:
            return object.__getattribute__(self,attr)

    @property
    def n_slices(self):
        return self._n_slices

    @property
    def z_bins(self):
        return self._z_bins

    @property
    def intensity(self):
        return self._intensity

    @property
    def n_macroparticles_per_slice(self):
        return self._intensity_distribution


    def slice_sets(self):
        return [self]

    def signal(self, axis = 'x'):

        parameters = Parameters(1, self._bin_edges, 1, self._n_slices,
                                [self._offset])

        if axis == 'x':
            parameters['location'] = self._location_x
            parameters['beta'] = self._beta_x
            signal = np.copy(self.x)
        elif axis == 'xp':
            parameters['location'] = self._location_x
            parameters['beta'] = self._beta_x
            signal = np.copy(self.xp)
        elif axis == 'y':
            parameters['location'] = self._location_y
            parameters['beta'] = self._beta_y
            signal = np.copy(self.y)
        elif axis == 'yp':
            parameters['location'] = self._location_y
            parameters['beta'] = self._beta_y
            signal = np.copy(self.yp)
        elif axis == 'z':
            signal = np.copy(self.z)
        elif axis == 'dp':
            signal = np.copy(self.dp)
        elif axis == 't':
            signal = np.copy(self.t)
        else:
            raise ValueError('Unknown axis')

        return parameters, signal

    def rotate(self, angle, axis='x'):
        s = np.sin(angle)
        c = np.cos(angle)

        if (axis == 'x') or (axis == 'xp'):
            self._total_angle_x += angle
            np.copyto(self.temp_x, c * self.x + self._beta_x * s * self.xp)
            np.copyto(self.temp_xp,(-1. / self._beta_x) * s * self.x + c * self.xp)
            np.copyto(self.x, self.temp_x)
            np.copyto(self.xp, self.temp_xp)
#             self.x = new_x
#             self.xp = new_xp
        elif (axis == 'y') or (axis == 'yp'):
            self._total_angle_y += angle
            new_y = c * self.y + self._beta_y * s * self.yp
            new_yp = (-1. / self._beta_y) * s * self.y + c * self.yp
            self.y = new_y
            self.yp = new_yp
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

    def fixed_amplitude(self, axis='x'):
        if axis == 'x':
            s = np.sin(-1. * self._total_angle_x)
            c = np.cos(-1. * self._total_angle_x)
            return c * self.x + self._beta_x * s * self.xp

        elif axis == 'y':
            s = np.sin(-1. * self._total_angle_y)
            c = np.cos(-1. * self._total_angle_y)
            return c * self.y + self._beta_y * s * self.yp

        elif axis == 'xp':
            s = np.sin(-1. * self._total_angle_x)
            c = np.cos(-1. * self._total_angle_x)
            return (-1. / self._beta_x) * s * self.x + c * self.xp

        elif axis == 'yp':
            s = np.sin(-1. * self._total_angle_x)
            c = np.cos(-1. * self._total_angle_x)
            return (-1. / self._beta_y) * s * self.y + c * self.yp

        else:
            raise ValueError('Unknown axis')


    def init_noise(self,noise_level, axis='x'):
        if axis == 'x':
            self.x = np.random.normal(0., noise_level, len(self.x))
        else:
            raise ValueError('Unknown axis')


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
