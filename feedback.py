import numpy as np
from processors import Register

def get_statistical_variables(processors, variables = None):
    """
    Function which checks statistical variables required by signal processors

    :param processors: a list of signal processors
    :param variables: a list of statistical variables determined earlier
    :return: a list of statistical variables, which is a sum of variables from input list and those found from
    the signal processors
    """

    if variables is None:
        variables = []

    for processor in processors:
        variables.extend(processor.required_variables)

    variables = list(set(variables))

    if 'z_bins' in variables:
        variables.remove('z_bins')

    if 'n_macroparticles_per_slice' in variables:
        variables.remove('n_macroparticles_per_slice')


    return variables


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
        slice_set = bunch.get_slices(self.slicer, statistics = ['mean_xp', 'mean_yp'])

        # Reads a particle index and a slice index for each macroparticle
        p_idx = slice_set.particles_within_cuts
        s_idx = slice_set.slice_index_of_particle.take(p_idx)

        bunch.xp[p_idx] -= self.gain * slice_set.mean_xp[s_idx]
        bunch.yp[p_idx] -= self.gain * slice_set.mean_yp[s_idx]


class OneboxFeedback(object):
    """ General class for a simple feedback, where a pick up and a kicker is located in the same place. It takes
        mean_xp/yp or mean_x/y values of each slice and pass them through signal processor chains given in parameters
        processors_x and processors_y. The final correction for x/y or xp/yp values is a gain times the signals through
        the signal processors. Axes (xp/yp or x/y) can be chosen by giving input parameter axis='divergence' for xp/yp
        and axis='displacement' for x/y. The default axis is divergence.
    """
    def __init__(self, gain, slicer, processors_x, processors_y, axis='divergence'):

        self.slicer = slicer
        self.gain = gain

        self.processors_x = processors_x
        self.processors_y = processors_y

        self.axis = axis

        self.statistical_variables = None

    def track(self,bunch):

        if self.statistical_variables is None:
            if self.axis == 'divergence':
                self.statistical_variables = ['mean_xp', 'mean_yp']
            elif self.axis == 'displacement':
                self.statistical_variables = ['mean_x', 'mean_y']

            self.statistical_variables = get_statistical_variables(self.processors_x, self.statistical_variables)
            self.statistical_variables = get_statistical_variables(self.processors_y, self.statistical_variables)

        slice_set = bunch.get_slices(self.slicer, statistics=self.statistical_variables)

        signal_x = np.array([])
        signal_y = np.array([])

        if self.axis == 'divergence':
            signal_x = np.array([s for s in slice_set.mean_xp])
            signal_y = np.array([s for s in slice_set.mean_yp])

        elif self.axis == 'displacement':
            signal_x = np.array([s for s in slice_set.mean_x])
            signal_y = np.array([s for s in slice_set.mean_y])

        for processor in self.processors_x:
            signal_x = processor.process(signal_x,slice_set)

        for processor in self.processors_y:
            signal_y = processor.process(signal_y,slice_set)

        correction_x = self.gain*signal_x
        correction_y = self.gain*signal_y

        p_idx = slice_set.particles_within_cuts
        s_idx = slice_set.slice_index_of_particle.take(p_idx)

        if self.axis == 'divergence':
            bunch.xp[p_idx] -= correction_x[s_idx]
            bunch.yp[p_idx] -= correction_y[s_idx]

        elif self.axis == 'displacement':
            bunch.x[p_idx] -= correction_x[s_idx]
            bunch.y[p_idx] -= correction_y[s_idx]


class PickUp(object):
    """ General class for a pickup. It takes mean_x and mean_y values of each slice and pass them through signal processor
        chains given in input parameters signal_processors_x and signal_processors_y. Note that the signals are
        stored only to registers in the signal processor chains!
    """
    def __init__(self,slicer,processors_x,processors_y):

        self.slicer = slicer

        self.processors_x = processors_x
        self.processors_y = processors_y

        self.signal_x = []
        self.signal_y = []

        self.statistical_variables = None

    def track(self,bunch):

        if self.statistical_variables is None:
            self.statistical_variables = ['mean_x', 'mean_y']
            self.statistical_variables = get_statistical_variables(self.processors_x, self.statistical_variables)
            self.statistical_variables = get_statistical_variables(self.processors_y, self.statistical_variables)

        slice_set = bunch.get_slices(self.slicer, statistics=self.statistical_variables)

        self.signal_x = np.array([s for s in slice_set.mean_x])
        self.signal_y = np.array([s for s in slice_set.mean_y])

        for processor in self.processors_x:
            self.signal_x = processor.process(self.signal_x,slice_set)

        for processor in self.processors_y:
            self.signal_y = processor.process(self.signal_y,slice_set)


class Kicker(object):
    """ General class for a kicker. It takes signals from variable number of registers given in lists registers_x and
        registers_y. The total signal is produced by combining those signals in a mixer object (input parameters
        signal_mixer_x and signal_mixer_y). The final kick signal is calculated by passing the total signal through
        a signal processor chain (input parameters signal_processors_x and signal_processors_y) and multiplying that
        with gain. In order to take into account betatron phase differences between registers and the kicker, betatron
        phase angles (from the reference point of the accelerator) in x and y plane must be given as a parameter
        (input parameters phase_angle_x and phase_angle_y).
    """

    def __init__(self,position_x,position_y,gain,slicer,registers_x,registers_y,processors_x,processors_y,signal_mixer_x,signal_mixer_y):

        self.gain=gain
        self.slicer = slicer

        self.position_x = position_x
        self.position_y = position_y

        self.registers_x = registers_x
        self.registers_y = registers_y

        self.processors_x = processors_x
        self.processors_y = processors_y

        self.signal_mixer_x = signal_mixer_x
        self.signal_mixer_y = signal_mixer_y

        self.statistical_variables = None

    def track(self,bunch):

        if self.statistical_variables is None:
            self.statistical_variables = ['mean_xp', 'mean_yp']
            self.statistical_variables = get_statistical_variables(self.processors_x, self.statistical_variables)
            self.statistical_variables = get_statistical_variables(self.processors_y, self.statistical_variables)

        slice_set = bunch.get_slices(self.slicer, statistics=self.statistical_variables)

        signal_x = self.signal_mixer_x.mix(self.registers_x,self.position_x)
        signal_y = self.signal_mixer_y.mix(self.registers_y,self.position_y)

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


class FIRRegister(Register):

    def __init__(self,delay, tune, avg_length, position=None, n_slices=None, in_processor_chain=True):
        self.type = 'plain'
        super(self.__class__, self).__init__(delay, tune, avg_length, position, n_slices, in_processor_chain)
        self.required_variables = []

    def combine(self,x1,x2,reader_position,x_to_xp = False):
        # determines a complex number representation from two signals (e.g. from two pickups or different turns), by using
        # knowledge about phase advance between signals. After this turns the vector to the reader's phase
        # TODO: Why not x2[3]-x1[3]?
        if (x1[3] is not None) and (x1[3] != x2[3]):
            phi_x1_x2 = x1[3]-x2[3]
        else:
            phi_x1_x2 = -1. * self.phase_shift_per_turn

        s = np.sin(phi_x1_x2)
        c = np.cos(phi_x1_x2)
        re = x1[0]
        im = (c*x1[0]-x2[0])/float(s)

        # turns the vector to the reader's position
        delta_phi = x1[2]
        if reader_position is not None:
            delta_position = x1[3] - reader_position
            delta_phi += delta_position
            if delta_position > 0:
                delta_phi -= self.phase_shift_per_turn
            if x_to_xp == True:
                delta_phi -= pi/2.

        s = np.sin(delta_phi)
        c = np.cos(delta_phi)

        return np.array([c*re-s*im,s*re+c*im])