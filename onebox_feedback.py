import numpy as np
from scipy.constants import c, e
import scipy.integrate as integrate
import scipy.special as special
import itertools

#TODO: check slicer get_slices vs extract_slices

class IdealBunchFeedback(object):
    # The simplest possible feedback which correct a mean xp value of the bunch.
    def __init__(self,gain):
        self.gain = gain    # fraction of offset is corrected each
        self.counter=0  # number of track calls

    def track(self,bunch):

        # change xp value
        bunch.xp -= self.gain*bunch.mean_xp()
        bunch.yp -= self.gain*bunch.mean_yp()

        self.counter +=1


class IdealSliceFeedback(object):
        # correct a mean xp value of each slice in the bunch.
    def __init__(self,gain,slicer):
        self.slicer = slicer
        self.gain = gain    # fraction of offset is corrected each
        _, self.n_slices, _, _=self.slicer.config

    def track(self,bunch):
        slice_set = bunch.get_slices(self.slicer, statistics=True)

        # read particle index and slice index for each macroparticle
        p_idx = slice_set.particles_within_cuts
        s_idx = slice_set.slice_index_of_particle.take(p_idx)

        for p_id, s_id in itertools.izip(p_idx,s_idx):
            bunch.xp[p_id] -= self.gain*slice_set.mean_xp[s_id]
            bunch.yp[p_id] -= self.gain*slice_set.mean_yp[s_id]


class MatrixSliceFeedback(object):
    def __init__(self,gain,slicer,FBmatrix):
        self.FBmatrix = FBmatrix
        self.slicer = slicer
        self.gain = gain
        self.counter=0
        self.mode, self.n_slices, _, _=slicer.config

    def track(self,bunch):
        slice_set = bunch.get_slices(self.slicer, statistics=['mean_xp', 'mean_yp','mean_z'])

        signal_xp = np.array([s for s in slice_set.mean_xp])
        signal_yp = np.array([s for s in slice_set.mean_yp])

        correction_xp = self.gain*np.dot(self.FBmatrix,signal_xp)
        correction_yp = self.gain*np.dot(self.FBmatrix,signal_yp)

        p_idx = slice_set.particles_within_cuts
        s_idx = slice_set.slice_index_of_particle.take(p_idx)

        for p_id, s_id in itertools.izip(p_idx,s_idx):
            bunch.xp[p_id] -= correction_xp[s_id]
            bunch.yp[p_id] -= correction_yp[s_id]


class PhaseLinFeedback(object):
    def __init__(self,gain,delay,slicer,cutoff_frequency):
        self.cutoff = cutoff_frequency  #cutoff frequency of 1st order low pass filter
        self.slicer = slicer
        self.gain = gain
        self.delay = delay  # delay between offset reading and correction [turns/track calls]
        self.counter=0
        self.norm_coeff=1
        self.mode, self.n_slices, _, _=slicer.config

        # register which holds xp/yp values during the delay
        self.xp_register = [np.zeros(self.n_slices) for x in range(self.delay+1)]
        self.yp_register = [np.zeros(self.n_slices) for x in range(self.delay+1)]

        #signal spread matrix
        self.FBmatrix = np.identity(self.n_slices)

    def slice_weight(self,z0,z1,z2):
        # The weight of a slice in FB matrix. The weight is calculated from analytical expression for step response of
        # a phase linearized 1st order low pass filter, which correspond to Modified Bessel function of second kind.
        # The argument of Bessel function is cutoff frequency of filter times distance in time (z position / speed of
        # light
        int_from=self.cutoff*(z1-z0)/c
        int_to=self.cutoff*(z2-z0)/c
        integral_value, _= integrate.quad(lambda y: special.k0(abs(y)), int_from , int_to)
        return integral_value/self.norm_coeff

    def calculate_norm_coeff(self,slice_set):
        # Normalization coefficient for slice weights. The coefficient correspond to area of weight function integrated
        # over bunch (origin is in the midpoint of macro particles).
        bunch_length=slice_set.z_bins[-1]-slice_set.z_bins[0]
        bunch_midpoint=(slice_set.z_bins[-1]+slice_set.z_bins[0])/2

        int_from=self.cutoff*(bunch_midpoint-bunch_length/2)/c
        int_to=self.cutoff*(bunch_midpoint+bunch_length/2)/c
        self.norm_coeff, _=integrate.quad(lambda y: special.k0(abs(y)),int_from, int_to)

    def track(self,bunch):
        slice_set = bunch.get_slices(self.slicer, statistics=['mean_xp', 'mean_yp','mean_z'])

        # in first function call or when slice spacing changes FBmatrix is calculated. FB matrix describes singal spread between slices
        if self.counter == 0 or self.mode != 'uniform_bin':
            self.calculate_norm_coeff(slice_set)

            for i, mean_z in enumerate(slice_set.mean_z):
                for j in range(self.n_slices):
                    self.FBmatrix[i][j]=self.slice_weight(mean_z,slice_set.z_bins[j],slice_set.z_bins[j+1])

        #format a raw signal, which is xp value of each slice
        raw_signal_xp=np.array([offset for offset in slice_set.mean_xp])
        raw_signal_yp=np.array([offset for offset in slice_set.mean_yp])

        # set correction signals to register. The correction signals are calculated from raw signals by multiplying
        # gain (scalar), FBmatrix (matrix) and raw_signal(vector)
        self.xp_register[(self.counter)%(self.delay+1)]=self.gain*np.dot(self.FBmatrix,raw_signal_xp)
        self.yp_register[(self.counter)%(self.delay+1)]=self.gain*np.dot(self.FBmatrix,raw_signal_yp)

        p_idx = slice_set.particles_within_cuts
        s_idx = slice_set.slice_index_of_particle.take(p_idx)

        # change xp and yp values of each macroparticle. The change corresponds correction signal values for each slice
        # in register
        for p_id, s_id in itertools.izip(p_idx,s_idx):
            bunch.xp[p_id] -= self.xp_register[(self.counter+1)%(self.delay+1)][s_id]
            bunch.yp[p_id] -= self.yp_register[(self.counter+1)%(self.delay+1)][s_id]

        self.counter +=1


class PhaseLinChargeWeightedFeedback(object):
    def __init__(self,gain,delay,slicer,cutoff_frequency):
        self.cutoff = cutoff_frequency  #cutoff frequency of 1st order low pass filter
        self.slicer = slicer
        self.gain = gain
        self.delay = delay  # delay between offset reading and correction [turns/track calls]
        self.counter=0
        self.norm_coeff=1
        self.mode, self.n_slices, _, _=slicer.config

        # register which holds xp/yp values during the delay
        self.xp_register = [np.zeros(self.n_slices) for x in range(self.delay+1)]
        self.yp_register = [np.zeros(self.n_slices) for x in range(self.delay+1)]

        #signal spread matrix
        self.FBmatrix = np.identity(self.n_slices)

    def slice_weight(self,z0,z1,z2):
        # The weight of a slice in FB matrix. The weight is calculated from analytical expression for step response of
        # a phase linearized 1st order low pass filter, which correspond to Modified Bessel function of second kind.
        # The argument of Bessel function is cutoff frequency of filter times distance in time (z position / speed of
        # light
        int_from=self.cutoff*(z1-z0)/c
        int_to=self.cutoff*(z2-z0)/c
        integral_value, _= integrate.quad(lambda y: special.k0(abs(y)), int_from , int_to)
        return integral_value/self.norm_coeff

    def calculate_norm_coeff(self,slice_set):
        # Normalization coefficient for slice weights. The coefficient correspond to area of weight function integrated
        # over bunch (origin is in the midpoint of macro particles).
        bunch_length=slice_set.z_bins[-1]-slice_set.z_bins[0]
        bunch_midpoint=(slice_set.z_bins[-1]+slice_set.z_bins[0])/2

        int_from=self.cutoff*(bunch_midpoint-bunch_length/2)/c
        int_to=self.cutoff*(bunch_midpoint+bunch_length/2)/c
        self.norm_coeff, _=integrate.quad(lambda y: special.k0(abs(y)),int_from, int_to)

    def track(self,bunch):
        slice_set = bunch.get_slices(self.slicer, statistics=['mean_xp', 'mean_yp','mean_z'])

        # in first function call or when slice spacing changes FBmatrix is calculated. FB matrix describes singal spread between slices
        if self.counter == 0 or self.mode != 'uniform_bin':
            self.calculate_norm_coeff(slice_set)

            for i , mean_z in enumerate(slice_set.mean_z):
                for j in range(self.n_slices):
                    self.FBmatrix[i][j]=self.slice_weight(mean_z,slice_set.z_bins[j],slice_set.z_bins[j+1])

        n_macroparticles = np.sum(slice_set.n_macroparticles_per_slice)

        #format a raw signal, which is charge normalized xp value of each slice
        raw_signal_xp=np.array([offset*strength for offset, strength in itertools.izip(slice_set.mean_xp, slice_set.n_macroparticles_per_slice)])*self.n_slices/n_macroparticles
        raw_signal_yp=np.array([offset*strength for offset, strength in itertools.izip(slice_set.mean_yp, slice_set.n_macroparticles_per_slice)])*self.n_slices/n_macroparticles

        # set correction signals to register. The correction signals are calculated from raw signals by multiplying
        # gain (scalar), FBmatrix (matrix) and raw_signal(vector)
        self.xp_register[(self.counter)%(self.delay+1)]=self.gain*np.dot(self.FBmatrix,raw_signal_xp)
        self.yp_register[(self.counter)%(self.delay+1)]=self.gain*np.dot(self.FBmatrix,raw_signal_yp)

        p_idx = slice_set.particles_within_cuts
        s_idx = slice_set.slice_index_of_particle.take(p_idx)

        # Change xp and yp values of each macroparticle. Changes corresponds to correction signal values for each slice
        # in register
        for p_id, s_id in itertools.izip(p_idx,s_idx):
            bunch.xp[p_id] -= self.xp_register[(self.counter+1)%(self.delay+1)][s_id]
            bunch.yp[p_id] -= self.yp_register[(self.counter+1)%(self.delay+1)][s_id]

        self.counter +=1