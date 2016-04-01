import numpy as np
import itertools

#TODO: check slicer get_slices vs extract_slices

class IdealBunchFeedback(object):
    """The simplest possible feedback which correct a mean xp value of the bunch."""
    def __init__(self,gain):
        self.gain = gain    # Fraction of offset is corrected in each turn

    def track(self,bunch):
        bunch.xp -= self.gain*bunch.mean_xp()
        bunch.yp -= self.gain*bunch.mean_yp()


class IdealSliceFeedback(object):
    """Correct mean xp value of each slice"""
    def __init__(self,gain,slicer):
        self.slicer = slicer
        self.gain = gain    # Fraction of offset is corrected each
        _, self.n_slices, _, _=self.slicer.config

    def track(self,bunch):
        slice_set = bunch.get_slices(self.slicer, statistics=True)

        # Read particle index and slice index for each macroparticle
        p_idx = slice_set.particles_within_cuts
        s_idx = slice_set.slice_index_of_particle.take(p_idx)

        # Change the position of each macroparticles. The change is proportional to mean xp/yp value of the slice
        for p_id, s_id in itertools.izip(p_idx,s_idx):
            bunch.xp[p_id] -= self.gain*slice_set.mean_xp[s_id]
            bunch.yp[p_id] -= self.gain*slice_set.mean_yp[s_id]


class OneboxFeedback(object):
    """A general feedback object, where pick up and kicker is located in same place."""
    def __init__(self,gain,slicer,signal_processors_x,signal_processors_y):
        self.slicer = slicer
        self.gain = gain
        self.signal_processors_x = signal_processors_x
        self.signal_processors_y = signal_processors_y
        self.mode, self.n_slices, _, _=slicer.config

    def track(self,bunch):
        """The signal (xp/yp values of slices) goes through all signal processors, which gives corections"""
        slice_set = bunch.get_slices(self.slicer, statistics=['mean_xp', 'mean_yp','mean_z'])

        signal_xp = np.array([s for s in slice_set.mean_xp])
        signal_yp = np.array([s for s in slice_set.mean_yp])

        for signal_processor in self.signal_processors_x:
            signal_xp = signal_processor.process(signal_xp,slice_set)

        for signal_processor in self.signal_processors_y:
            signal_yp = signal_processor.process(signal_yp,slice_set)

        correction_xp = self.gain*signal_xp
        correction_yp = self.gain*signal_yp

        p_idx = slice_set.particles_within_cuts
        s_idx = slice_set.slice_index_of_particle.take(p_idx)

        for p_id, s_id in itertools.izip(p_idx,s_idx):
            bunch.xp[p_id] -= correction_xp[s_id]
            bunch.yp[p_id] -= correction_yp[s_id]

