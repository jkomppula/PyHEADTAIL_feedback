import numpy as np
from scipy.constants import c

from ..core import Parameters

class Bunch(object):
    def __init__(self, length, charge=0, n_slices=100, location_x=0., location_y=0.,
                 beta_x=1., beta_y=1., offset=0., distribution='KV'):
        self._length = length
        self._charge = charge
        self._n_slices = n_slices
        self._beta_x = beta_x
        self._beta_y = beta_y
        self._location_x = location_x
        self._location_y = location_y
        self._offset = offset
        self._distribution = distribution

        self._z_bins = np.linspace(-1.*c*length/2., c*length/2., self._n_slices+1) + self._offset
        self._bin_edges = np.transpose(np.array([self._z_bins[:-1], self._z_bins[1:]]))

        self._z = (self._bin_edges[:, 0]+self._bin_edges[:, 1])/2.
        self._x = np.zeros(self._n_slices)
        self._xp = np.zeros(self._n_slices)
        self._y = np.zeros(self._n_slices)
        self._yp = np.zeros(self._n_slices)
        self._dp = np.zeros(self._n_slices)

        self._t = self._z/c

        self._total_angle_x = 0.
        self._total_angle_y = 0.

        if self._distribution == 'KV':
            self._charge_distribution = np.ones(self._n_slices)*self._charge/float(self._n_slices)
        elif self._distribution == 'waterbag':
            self._charge_distribution = (1.-self._z*self._z/(self._length*self._length))**2
            norm_coeff = self._charge/np.sum(self._charge_distribution)
            self._charge_distribution = self._charge_distribution/norm_coeff
#        if self._distribution == 'parabolic':
#            pass
        else:
            raise ValueError('Unknown distribution!')

    @property
    def n_slices(self):
        return self._n_slices

    @property
    def z_bins(self):
        return self._z_bins

    @property
    def n_macroparticles_per_slice(self):
        return self._charge_distribution

    @property
    def mean_x(self):
        return self._x

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self._x = value

    @property
    def x_amp(self):
        return self.normalized_amplitude('x')

    @property
    def x_fixed(self):
        return self.fixed_amplitude('x')

    @property
    def mean_y(self):
        return self._y

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value):
        self._y = value

    @property
    def y_amp(self):
        return self.normalized_amplitude('y')

    @property
    def y_fixed(self):
        return self.fixed_amplitude('y')

    @property
    def mean_z(self):
        return self._z

    @property
    def z(self):
        return self._z

    @z.setter
    def z(self, value):
        self._z = value

    @property
    def mean_xp(self):
        return self._xp

    @property
    def xp(self):
        return self._xp

    @xp.setter
    def xp(self, value):
        self._xp = value

    @property
    def xp_amp(self):
        return self.normalized_amplitude('xp')

    @property
    def xp_fixed(self):
        return self.fixed_amplitude('xp')

    @property
    def mean_yp(self):
        return self._yp

    @property
    def yp(self):
        return self._yp

    @yp.setter
    def yp(self, value):
        self._yp = value

    @property
    def yp_amp(self):
        return self.normalized_amplitude('yp')

    @property
    def yp_fixed(self):
        return self.fixed_amplitude('yp')

    @property
    def mean_dp(self):
        return self._dp

    @property
    def dp(self):
        return self._dp

    @dp.setter
    def dp(self, value):
        self._dp = value

    def slice_sets(self):
        return [self]

    def signal(self, axis = 'x'):

        parameters = Parameters(1, self._bin_edges, 1, self._n_slices,
                                [self._offset])

        if axis == 'x':
            parameters['location'] = self._location_x
            parameters['beta'] = self._beta_x
            signal = np.copy(self._x)
        elif axis == 'xp':
            parameters['location'] = self._location_x
            parameters['beta'] = self._beta_x
            signal = np.copy(self._xp)
        elif axis == 'y':
            parameters['location'] = self._location_y
            parameters['beta'] = self._beta_y
            signal = np.copy(self._y)
        elif axis == 'yp':
            parameters['location'] = self._location_y
            parameters['beta'] = self._beta_y
            signal = np.copy(self._yp)
        elif axis == 'z':
            signal = np.copy(self._z)
        elif axis == 'dp':
            signal = np.copy(self._dp)
        elif axis == 't':
            signal = np.copy(self._t)
        else:
            raise ValueError('Unknown axis')

        return parameters, signal

    def rotate(self, angle, axis='x'):
        s = np.sin(angle)
        c = np.cos(angle)

        if (axis == 'x') or (axis == 'xp'):
            self._total_angle_x += angle
            new_x = c * self._x + self._beta_x * s * self._xp
            new_xp = (-1. / self._beta_x) * s * self._x + c * self._xp
            self._x = new_x
            self._xp = new_xp
        elif (axis == 'y') or (axis == 'yp'):
            self._total_angle_y += angle
            new_y = c * self._y + self._beta_y * s * self._yp
            new_yp = (-1. / self._beta_y) * s * self._y + c * self._yp
            self._y = new_y
            self._yp = new_yp
        else:
            raise ValueError('Unknown axis')

    def normalized_amplitude(self, axis='x'):
        if axis == 'x':
            return np.sqrt(self._x**2 + (self._beta_x*self._xp)**2)
        elif axis == 'y':
            return np.sqrt(self._y**2 + (self._beta_y*self._yp)**2)
        elif axis == 'xp':
            return np.sqrt(self._xp**2 + (self._x/self._beta_x)**2)
        elif axis == 'yp':
            return np.sqrt(self._yp**2 + (self._y/self._beta_y)**2)
        else:
            raise ValueError('Unknown axis')

    def fixed_amplitude(self, axis='x'):
        if axis == 'x':
            s = np.sin(-1. * self._total_angle_x)
            c = np.cos(-1. * self._total_angle_x)
            return c * self._x + self._beta_x * s * self._xp

        elif axis == 'y':
            s = np.sin(-1. * self._total_angle_y)
            c = np.cos(-1. * self._total_angle_y)
            return c * self._y + self._beta_y * s * self._yp

        elif axis == 'xp':
            s = np.sin(-1. * self._total_angle_x)
            c = np.cos(-1. * self._total_angle_x)
            return (-1. / self._beta_x) * s * self._x + c * self._xp

        elif axis == 'yp':
            s = np.sin(-1. * self._total_angle_x)
            c = np.cos(-1. * self._total_angle_x)
            return (-1. / self._beta_y) * s * self._y + c * self._yp

        else:
            raise ValueError('Unknown axis')


class Beam(object):
    def __init__(self, filling_scheme, circumference, h_RF, bunch_length, **kwargs):
        self._filling_scheme = sorted(filling_scheme)
        self._circumference = circumference
        self._h_RF = h_RF

        self._offsets = np.zeros(len(filling_scheme))

        for i, bucket in enumerate(filling_scheme):
            self._offsets[i] = float(bucket)*self._circumference/float(self._h_RF)

        self._bunch_list = []
        for offset in self._offsets:
            self._bunch_list.append(Bunch(bunch_length, offset=offset, **kwargs))

        self._n_slices_per_bunch = self._bunch_list[0].n_slices

    @property
    def n_slices_per_bunch(self):
        return self._n_slices_per_bunch

    @property
    def n_macroparticles_per_slice(self):
        return self._charge_distribution

    @property
    def x(self):
        return self.combine_property('x')

    @x.setter
    def x(self, values):
        self.__set_values(values, 'x')

    @property
    def x_amp(self):
        return self.combine_property('x_amp')

    @property
    def x_fixed(self):
        return self.combine_property('x_fixed')

    @property
    def y(self):
        return self.combine_property('y')

    @y.setter
    def y(self, values):
        self.__set_values(values, 'y')

    @property
    def y_amp(self):
        return self.combine_property('y_amp')

    @property
    def y_fixed(self):
        return self.combine_property('y_fixed')

    @property
    def z(self):
        return self.combine_property('z')

    @z.setter
    def z(self, values):
        self.__set_values(values, 'z')

    @property
    def z_amp(self):
        return self.combine_property('z_amp')

    @property
    def z_fixed(self):
        return self.combine_property('z_fixed')

    @property
    def xp(self):
        return self.combine_property('xp')

    @xp.setter
    def xp(self, values):
        self.__set_values(values, 'xp')

    @property
    def xp_amp(self):
        return self.combine_property('xp_amp')

    @property
    def xp_fixed(self):
        return self.combine_property('xp_fixed')

    @property
    def yp(self):
        return self.combine_property('yp')

    @yp.setter
    def yp(self, values):
        self.__set_values(values, 'yp')

    @property
    def yp_amp(self):
        return self.combine_property('yp_amp')

    @property
    def yp_fixed(self):
        return self.combine_property('yp_fixed')

    @property
    def dp(self):
        return self.combine_property('dp')

    @dp.setter
    def dp(self, values):
        self.__set_values(values, 'dp')


    @property
    def slice_sets(self):
        return self._bunch_list

    def signal(self, var = 'x'):
        signal = None
        parameters = None
        bin_edges = None
        total_length = 0

        for i, bunch in enumerate(self._bunch_list):
            temp_parameters, temp_signal = bunch.signal(var)

            if parameters is None:
                parameters = temp_parameters
                parameters['n_segments'] = len(self._bunch_list)
                parameters['n_bins_per_segment'] = len(temp_signal)
                parameters['segment_midpoints'] = self._offsets
                total_length = parameters['n_segments'] * parameters['n_bins_per_segment']

            if signal is None:
                signal = np.zeros(total_length)

            if bin_edges is None:
                bin_edges = np.zeros((total_length, 2))

            i_from = i*parameters['n_bins_per_segment']
            i_to = (i + 1)*parameters['n_bins_per_segment']

            np.copyto(signal[i_from:i_to], temp_signal)
            np.copyto(bin_edges[i_from:i_to, :], temp_parameters['bin_edges'])

        parameters['bin_edges'] = bin_edges

        return parameters, signal

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

        return combined

    def __set_values(self, values, axis='x'):
        for i, bunch in enumerate(self._bunch_list):
            i_from = i * self._n_slices_per_bunch
            i_to = (i + 1) * self._n_slices_per_bunch

            setattr(bunch, axis, values[i_from:i_to])


    def rotate(self, angle, var='x'):

        for bunch in self._bunch_list:
            bunch.rotate(angle, var)


class SimpleBeam(Beam):
    def __init__(self, n_bunches, bunch_spacing, bunch_length, **kwargs):
        self._n_bunches = n_bunches
        self._bunch_spacing = bunch_spacing
        self._bunch_length = bunch_length

        filling_scheme = np.arange(0, n_bunches)
        h_RF = 10*n_bunches
        circumference = h_RF * bunch_spacing*c
        super(self.__class__, self).__init__(filling_scheme, circumference,
                                             h_RF, bunch_length, **kwargs)



