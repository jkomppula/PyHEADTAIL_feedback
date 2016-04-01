import numpy as np
import itertools
import math


class PickUp(object):
    def __init__(self,slicer,signal_processors_x,signal_processors_y,phase_shift):
        """Takes x/y values of slices and pass them through signal processors. The signal processors handle all
        necessary operations including registers/averaging, phase shifting, etc"""
        self.slicer = slicer

        self.signal_processors_x = signal_processors_x
        self.signal_processors_y = signal_processors_y

        self.phase_shift = phase_shift # place of the pick up in radians

        self.signal_x = []
        self.signal_y = []

    def track(self,bunch):
        slice_set = bunch.get_slices(self.slicer, statistics=['mean_x', 'mean_y','mean_z'])

        self.signal_x = np.array([s for s in slice_set.mean_x])
        self.signal_y = np.array([s for s in slice_set.mean_y])

        for signal_processor in self.signal_processors_x:
            self.signal_x = signal_processor.process(self.signal_x,slice_set)

        for signal_processor in self.signal_processors_y:
            self.signal_y = signal_processor.process(self.signal_y,slice_set)


# TODO: Add pi/2 phase shift correction between pick up and kicker
class Kicker(object):
    """Combines signals from different pick ups by using signal_mixer object. After this the signals pass through
    signal processor chains, which produce final correction signals"""

    def __init__(self,gain,phase_shift,slicer,pickups,signal_processors_x,signal_processors_y,signal_mixer_x,signal_mixer_y):
        self.gain=gain
        self.phase_shift = phase_shift
        self.pickups=pickups    # list of pick ups
        self.signal_processors_x = signal_processors_x
        self.signal_processors_y = signal_processors_y
        self.signal_mixer_x = signal_mixer_x
        self.signal_mixer_y = signal_mixer_y

        self.slicer = slicer
        self.mode, self.n_slices, _, _=slicer.config

    def track(self,bunch):
        slice_set = bunch.get_slices(self.slicer, statistics=['mean_xp', 'mean_yp','mean_z'])

        signal_x = self.signal_mixer_x.mix(self.phase_shift,self.pickups)
        signal_y = self.signal_mixer_x.mix(self.phase_shift,self.pickups)

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

# TODO: Check vector sum of complex numbers
class AverageMixer(object):
    """The simplest possible signal mixer of pick ups, which calculates a phase weighted average of
    the pick up signals"""
    def __init__(self,channel):
        self.channel = channel

    def mix(self,kicker_phase_shift,pickups):
        signal = None

        for index, pickup in enumerate(pickups):
            if signal is None:
                signal = np.zeros(len(pickup.signal_x))

            if self.channel == 'x':
                signal += math.cos(pickup.phase_shift-kicker_phase_shift)*pickup.signal_x/len(pickups)
            elif self.channel == 'y':
                signal += math.cos(pickup.phase_shift-kicker_phase_shift)*pickup.signal_y/len(pickups)
