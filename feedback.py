import numpy as np



class IdealBunchFeedback(object):
    """ The simplest possible feedback. It corrects a gain fraction of a mean xp/yp value of the bunch.
    """
    def __init__(self,gain):

        self.gain = gain

    def track(self,bunch):
        bunch.xp -= self.gain*bunch.mean_xp()
        bunch.yp -= self.gain*bunch.mean_yp()


class IdealSliceFeedback(object):
    """Corrects a gain fraction of a mean xp/yp value of each slice in the bunch."""
    def __init__(self,gain,slicer):

        self.slicer = slicer
        self.gain = gain

    def track(self,bunch):
        slice_set = bunch.get_slices(self.slicer, statistics=True)

        # Reads a particle index and a slice index for each macroparticle
        p_idx = slice_set.particles_within_cuts
        s_idx = slice_set.slice_index_of_particle.take(p_idx)

        bunch.xp[p_idx] -= self.gain * slice_set.mean_xp[s_idx]
        bunch.yp[p_idx] -= self.gain * slice_set.mean_yp[s_idx]


class OneboxFeedback(object):
    """ General class for a simple feedback, where a pick up and a kicker is located in same place. It takes mean_xp and
        mean_yp values of slices and pass them through signal processor chains given in parameters processors_x
        and processors_y. The final correction for xp/yp value of each slice is a gain times the signals through
        the signal processors.
    """
    def __init__(self, gain, slicer, processors_x, processors_y):

        self.slicer = slicer
        self.gain = gain

        self.processors_x = processors_x
        self.processors_y = processors_y

    def track(self,bunch):
        slice_set = bunch.get_slices(self.slicer, statistics=['mean_xp', 'mean_yp','mean_z'])

        signal_xp = np.array([s for s in slice_set.mean_xp])
        signal_yp = np.array([s for s in slice_set.mean_yp])

        for processor in self.processors_x:
            signal_xp = processor.process(signal_xp,slice_set)

        for processor in self.processors_y:
            signal_yp = processor.process(signal_yp,slice_set)

        correction_xp = self.gain*signal_xp
        correction_yp = self.gain*signal_yp

        # Reads a particle index and a slice index for each macroparticle
        p_idx = slice_set.particles_within_cuts
        s_idx = slice_set.slice_index_of_particle.take(p_idx)

        bunch.xp[p_idx] -= correction_xp[s_idx]
        bunch.yp[p_idx] -= correction_yp[s_idx]


class PickUp(object):
    """ General class for a pickup. It takes mean_x and mean_y values of slices and pass them through signal processor
        chains given in input parameters signal_processors_x and signal_processors_y. Note that the signals are
        stored only to registers in the signal processor chains!
    """
    def __init__(self,slicer,processors_x,processors_y):

        self.slicer = slicer

        self.processors_x = processors_x
        self.processors_y = processors_y

        self.signal_x = []
        self.signal_y = []

    def track(self,bunch):
        slice_set = bunch.get_slices(self.slicer, statistics=['mean_x', 'mean_y','mean_z'])

        self.signal_x = np.array([s for s in slice_set.mean_x])
        self.signal_y = np.array([s for s in slice_set.mean_y])

        for processor in self.processors_x:
            self.signal_x = processor.process(self.signal_x,slice_set)

        for processor in self.processors_y:
            self.signal_y = processor.process(self.signal_y,slice_set)


class Kicker(object):
    """ General class for a kicker. It takes signals from variable number of registers given in lists registers_x and
        register_y. The total signal is produced by combining those signals in a mixer object (input parameters
        signal_mixer_x and signal_mixer_y). The final kick signal is calculated by passing the total signal through
        a signal processor chain (input parameters signal_processors_x and signal_processors_y) and multiplying that
        with gain. In order to take into account betatron phase differences between registers and the kicker, betatron
        phase angles (from the reference point of the accelerator) in x and y plane must be given as a parameter
        (input parameters phase_angle_x and phase_angle_y).
    """

    def __init__(self,phase_angle_x,phase_angle_y,gain,slicer,registers_x,registers_y,processors_x,processors_y,signal_mixer_x,signal_mixer_y):

        self.gain=gain
        self.slicer = slicer

        self.phase_angle_x = phase_angle_x
        self.phase_angle_y = phase_angle_y

        self.registers_x = registers_x
        self.registers_y = registers_y

        self.processors_x = processors_x
        self.processors_y = processors_y

        self.signal_mixer_x = signal_mixer_x
        self.signal_mixer_y = signal_mixer_y

    def track(self,bunch):

        slice_set = bunch.get_slices(self.slicer, statistics=['mean_xp', 'mean_yp','mean_z'])

        signal_x = self.signal_mixer_x.mix(self.registers_x,self.phase_angle_x)
        signal_y = self.signal_mixer_y.mix(self.registers_y,self.phase_angle_y)

        for processor in self.processors_x:
            signal_x = processor.process(signal_x,slice_set)

        for processor in self.processors_y:
            signal_y = processor.process(signal_y,slice_set)

        correction_xp = self.gain*signal_x
        correction_yp = self.gain*signal_y

        # Reads a particle index and a slice index for each macroparticle
        p_idx = slice_set.particles_within_cuts
        s_idx = slice_set.slice_index_of_particle.take(p_idx)

        bunch.xp[p_idx] -= correction_xp[s_idx]
        bunch.yp[p_idx] -= correction_yp[s_idx]

