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

        self.z = (self._bin_edges[:, 0]+self._bin_edges[:, 1])/2.
        self.x = np.zeros(self._n_slices)
        self.xp = np.zeros(self._n_slices)
        self.y = np.zeros(self._n_slices)
        self.yp = np.zeros(self._n_slices)
        self.dp = np.zeros(self._n_slices)

        self.t = self.z/c

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
    def n_macroparticles_per_slice(self):
        return self._charge_distribution


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
            new_x = c * self.x + self._beta_x * s * self.xp
            new_xp = (-1. / self._beta_x) * s * self.x + c * self.xp
            self.x = new_x
            self.xp = new_xp
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

    def __getattr__(self,attr):
        if (attr in ['x','y','z','xp','yp','dp','x_amp','y_amp','z_amp','xp_amp','yp_amp','dp_amp', 'x_fixed','y_fixed','xp_fixed','yp_fixed']):
            return self.combine_property(attr)
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

    @property
    def n_macroparticles_per_slice(self):
        return self._charge_distribution

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



