import numpy as np
import itertools
import math

class PickUp(object):
    def __init__(self,slicer,signal_processors_x,signal_processors_y,phase_shift):
        self.slicer = slicer
        self.mode, self.n_slices, _, _=slicer.config

        self.signal_processors_x = signal_processors_x
        self.signal_processors_y = signal_processors_y
        self.phase_shift = phase_shift

        self.signal_x = []
        self.signal_y = []

    def track(self,bunch):
        slice_set = bunch.get_slices(self.slicer, statistics=['mean_xp', 'mean_yp','mean_z'])

        self.signal_x = np.array([s for s in slice_set.mean_x])
        self.signal_y = np.array([s for s in slice_set.mean_y])

        for signal_processor in self.signal_processors_x:
            self.signal_x = signal_processor.process(self.signal_x,slice_set)

        for signal_processor in self.signal_processors_y:
            self.signal_y = signal_processor.process(self.signal_y,slice_set)

class Kicker(object):
    def __init__(self,gain,phase_shift,slicer,pickups,signal_processors_x,signal_processors_y,pickup_signal_processors_x=None,pickup_signal_processors_y=None):
        self.gain=gain
        self.phase_shift = phase_shift
        self.pickups=pickups
        self.signal_processors_x = signal_processors_x
        self.signal_processors_y = signal_processors_y
        self.pickup_signal_processors_x = pickup_signal_processors_x
        self.pickup_signal_processors_y = pickup_signal_processors_y
        self.slicer = slicer
        self.mode, self.n_slices, _, _=slicer.config

    def track(self,bunch):
        slice_set = bunch.get_slices(self.slicer, statistics=['mean_xp', 'mean_yp','mean_z'])

        signal_x = None
        signal_y = None

        for index, pickup in enumerate(self.pickups):
            if signal_x is None:
                signal_x = np.zeros(len(pickup.signal_x))
            if signal_y is None:
                signal_y = np.zeros(len(pickup.signal_y))

            signal_x += math.cos(pickup.phase_shift-self.phase_shift)*pickup.signal_x/len(self.pickups)
            signal_y += math.cos(pickup.phase_shift-self.phase_shift)*pickup.signal_y/len(self.pickups)

        for signal_processor in self.signal_processors_x:
            signal_x = signal_processor.process(signal_x,slice_set)

        for signal_processor in self.signal_processors_y:
            signal_y = signal_processor.process(signal_y,slice_set)

        correction_xp = self.gain*signal_x
        correction_yp = self.gain*signal_y

        p_idx = slice_set.particles_within_cuts
        s_idx = slice_set.slice_index_of_particle.take(p_idx)

        for p_id, s_id in itertools.izip(p_idx,s_idx):
            bunch.xp[p_id] -= correction_xp[s_id]
            bunch.yp[p_id] -= correction_yp[s_id]

