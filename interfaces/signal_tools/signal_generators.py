import numpy as np
from scipy.constants import c

from ..core import Parameters


class SliceObject(object):
    def __init__(self, bin_edges, intensity, location_x=0, location_y=0, beta_x=1, beta_y=1,
                 x=None, xp=None, y=None, yp=None, dp=None,
                 bunch_id=None, circumference=None, h_RF=None, circular_overlapping = 0):

        self._bunch_id = bunch_id
        self._circumference = circumference
        self._h_RF = h_RF
        self._circular_overlapping = circular_overlapping

        self._location_x = location_x
        self._location_y = location_y
        self._beta_x = beta_x
        self._beta_y = beta_y

        self._bin_edges = bin_edges
        self._z = (self._bin_edges[:, 0]+self._bin_edges[:, 1])/2.
        self._z_bins = np.append(self._bin_edges[:, 0], np.array([self._bin_edges[-1, 1]]))
        self._n_slices = len(self._z)

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

    @property
    def bunc_id(self):
        return self._bunc_id

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
    def bin_edges(self):
        return self._bin_edges

    @property
    def n_macroparticles_per_slice(self):
        return self._intensity_distribution


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


    def signal(self, var = 'x'):
        if self._output_signal is None:
            self._output_signal = np.zeros(self._n_slices + 2 * self._circular_overlapping)

        if self._output_parameters is None:
            if self._circular_overlapping > 0:

                if self._bunch_id is not None:
                    bunch_mid = self.bunc_id * self._circumference/self._h_RF
                else:
                    bunch_mid = np.mean(self._z_bins)

                prefix_offset = self._bin_edges[self._circular_overlapping,0] - self._bin_edges[0,0]
                postfix_offset = self._bin_edges[-1,1] - self._bin_edges[-self._circular_overlapping,1]
                bin_edges = np.concatenate((self._bin_edges[0:self._circular_overlapping]+prefix_offset,self._bin_edges),axis=0)
                bin_edges = np.concatenate((bin_edges,self._bin_edges[-self._circular_overlapping:]+postfix_offset),axis=0)

                bin_edges = bin_edges + bunch_mid


                self._output_parameters = Parameters(2, bin_edges, 1, len(bin_edges),
                        [bunch_mid],location=self._location_x, beta=self._beta_x)

        if var == 'x':
            np.copyto(self._output_signal[self._circular_overlapping:
                (self._circular_overlapping+self._n_slices)], self.x)
        elif var == 'xp':
            np.copyto(self._output_signal[self._circular_overlapping:
                (self._circular_overlapping+self._n_slices)], self.xp)
        elif var == 'y':
            np.copyto(self._output_signal[self._circular_overlapping:
                (self._circular_overlapping+self._n_slices)], self.y)
        elif var == 'yp':
            np.copyto(self._output_signal[self._circular_overlapping:
                (self._circular_overlapping+self._n_slices)], self.yp)
        else:
            raise ValueError('Unknown axis')

        if self._circular_overlapping > 0 :
            np.copyto(self._output_signal[:self._circular_overlapping],
                      self._output_signal[self._n_slices:(self._circular_overlapping+self._n_slices)])
            np.copyto(self._output_signal[-self._circular_overlapping:],
                      self._output_signal[self._circular_overlapping:(2*self._circular_overlapping)])

        return self._output_parameters, self._output_signal

    def correction(self,signal, beam_map=None, var='x'):

        proper_signal = signal[self._circular_overlapping:(self._circular_overlapping+len(self._z))]
        if beam_map is None:
            beam_map = np.ones(len(self._z))

        if var == 'x':
            self.x[self._beam_map] = self.x[self._beam_map] - proper_signal[self._beam_map]
        elif var == 'xp':
            self.xp[self._beam_map] = self.xp[self._beam_map] - proper_signal[self._beam_map]
        elif var == 'y':
            self.y[self._beam_map] = self.y[self._beam_map] - proper_signal[self._beam_map]
        elif var == 'yp':
            self.yp[self._beam_map] = self.yp[self._beam_map] - proper_signal[self._beam_map]
        else:
            raise ValueError('Unknown axis')

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

    def fixed_amplitude(self, axis='x', offset_angle = 0.):
        if axis == 'x':
            s = np.sin(-1. * self._total_angle_x + offset_angle)
            c = np.cos(-1. * self._total_angle_x + offset_angle)
            return c * self.x + self._beta_x * s * self.xp

        elif axis == 'y':
            s = np.sin(-1. * self._total_angle_y + offset_angle)
            c = np.cos(-1. * self._total_angle_y + offset_angle)
            return c * self.y + self._beta_y * s * self.yp

        elif axis == 'xp':
            s = np.sin(-1. * self._total_angle_x + offset_angle)
            c = np.cos(-1. * self._total_angle_x + offset_angle)
            return (-1. / self._beta_x) * s * self.x + c * self.xp

        elif axis == 'yp':
            s = np.sin(-1. * self._total_angle_x + offset_angle)
            c = np.cos(-1. * self._total_angle_x + offset_angle)
            return (-1. / self._beta_y) * s * self.y + c * self.yp

        else:
            raise ValueError('Unknown axis')

    def init_noise(self,noise_level, var='x'):
        if var == 'x':
            self.x = np.random.normal(0., noise_level, len(self.x))
            self.xp = (1. / self._beta_x)*np.random.normal(0., noise_level, len(self.x))
        else:
            raise ValueError('Unknown variable')


class Bunch(SliceObject):
    def __init__(self, length, n_slices, intensity, distribution='KV', **kwargs):
        self.length = length
        distribution = distribution
        z_bins = np.linspace(-1.*c*length/2., c*length/2., n_slices)
        bin_edges = np.transpose(np.array([z_bins[:-1], z_bins[1:]]))
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

        self._bunch_list = []
        for bunch_id in self._filling_scheme:
            self._bunch_list.append(Bunch(bunch_length, n_slices, intensity,bunch_id=bunch_id,
                                          circumference=circumference, h_RF=h_RF,  **kwargs))

        self._n_slices_per_bunch = n_slices

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

        self._output_signal = None
        self._output_parameters = None


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
        if self._output_parameters is None:
            bin_edges = None
            beta = None
            location = None
            segment_midpoins = []
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
                    bin_edges = np.concatenate((bin_edges,parameters['bin_edges']),axis=0)

                segment_midpoins.append(parameters['segment_midpoints'][0])


        if self._output_parameters is None:
            self._output_parameters = Parameters(signal_class=1, bin_edges=bin_edges,
                                                 n_segments=n_segments,
                                                 n_bins_per_segment=n_bins_per_segment,
                                                 segment_midpoints=segment_midpoins,
                                                 location=location,
                                                 beta=beta)

        return self._output_parameters, self._output_signal

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
    def __init__(self, n_bunches, bunch_spacing, bunch_length, intensity, n_slices, **kwargs):
        self._n_bunches = n_bunches
        self._bunch_spacing = bunch_spacing

        filling_scheme = np.arange(0, n_bunches)
        h_RF = 10*n_bunches
        circumference = h_RF * bunch_spacing*c
        super(self.__class__, self).__init__(filling_scheme, circumference,
                                             h_RF, bunch_length, intensity, n_slices, **kwargs)


class CircularPointBeam(SliceObject):
    def __init__(self, filling_scheme, circumference, h_RF, intensity, circular_overlapping,
                 **kwargs):

        z_bins = np.linspace(0,circumference,h_RF)
        bin_edges =np.transpose(np.array([z_bins[:-1],
                                           z_bins[1:]]))
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

        super(self.__class__, self).__init__(bin_edges, intensities, bunch_id=0,
             circumference=circumference, h_RF=h_RF, circular_overlapping=circular_overlapping,
             **kwargs)

    def correction(self,signal, var='x'):
        super(CircularPointBeam, self).correction(signal, self._beam_map, var)


class SimpleCircularPointBeam(CircularPointBeam):
    def __init__(self, n_bunches, bunch_spacing, intensity, circular_overlapping, **kwargs):
        filling_scheme = np.arange(0, n_bunches)
        h_RF = n_bunches
        circumference = h_RF * bunch_spacing*c

        super(SimpleCircularPointBeam, self).__init__(filling_scheme, circumference, h_RF,
             intensity, circular_overlapping, **kwargs)
