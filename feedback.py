import numpy as np
import collections
from PyHEADTAIL_MPI.mpi import mpi_data
from core import get_processor_variables, process, Parameters
from processors.register import VectorSumCombiner, CosineSumCombiner
from processors.register import HilbertCombiner
"""
    This file contains modules, which can be used as a feedback module/object in PyHEADTAIL. Actual signal processing is
    done by using signal processors written to files processors.py and digital_processors.py. A list of signal
    processors is given as a argument for feedback elements.

    @author Jani Komppula
    @date 16/09/2016
    @copyright CERN
"""

"""
    Must be discussed:
        - turn by turn varying slice width -> will be forgot
        - future of matrix filters?
"""


class IdealBunchFeedback(object):
    """ The simplest possible feedback. It corrects a gain fraction of a mean xp/yp value of the bunch.
    """
    def __init__(self,gain):
        if isinstance(gain, collections.Container):
            self._gain_x = gain[0]
            self._gain_y = gain[1]
        else:
            self._gain_x = gain
            self._gain_y = gain

    def track(self,bunch):
        bunch.xp -= self._gain_x *bunch.mean_xp()
        bunch.yp -= self._gain_y*bunch.mean_yp()


class IdealSliceFeedback(object):
    """Corrects a gain fraction of a mean xp/yp value of each slice in the bunch."""
    def __init__(self,gain,slicer):
        if isinstance(gain, collections.Container):
            self._gain_x = gain[0]
            self._gain_y = gain[1]
        else:
            self._gain_x = gain
            self._gain_y = gain

        self._slicer = slicer

    def track(self,bunch):
        slice_set = bunch.get_slices(self._slicer, statistics = ['mean_xp', 'mean_yp'])

        # Reads a particle index and a slice index for each macroparticle
        p_idx = slice_set.particles_within_cuts
        s_idx = slice_set.slice_index_of_particle.take(p_idx)

        bunch.xp[p_idx] -= self._gain_x * slice_set.mean_xp[s_idx]
        bunch.yp[p_idx] -= self._gain_y * slice_set.mean_yp[s_idx]


def get_local_slice_sets(bunch, slicer, required_variables):
    signal_slice_sets = bunch.get_slices(slicer, statistics=required_variables)
    bunch_slice_sets = signal_slice_sets
    bunch_list = [bunch]

    return signal_slice_sets, bunch_slice_sets, bunch_list


def get_mpi_slice_sets(superbunch, mpi_gatherer):
    mpi_gatherer.gather(superbunch)
    signal_slice_sets = mpi_gatherer.bunch_by_bunch_data
    bunch_slice_sets = mpi_gatherer.slice_set_list
    bunch_list = mpi_gatherer.bunch_list

    return signal_slice_sets, bunch_slice_sets, bunch_list


def generate_parameters(signal_slice_sets, location=0., beta=1.):

    bin_edges = None
    segment_midpoints = []

    for slice_set in signal_slice_sets:
            edges = np.transpose(np.array([slice_set.z_bins[:-1],
                                           slice_set.z_bins[1:]]))
            segment_midpoints.append(np.mean(slice_set.z_bins))
            if bin_edges is None:
                bin_edges = np.copy(edges)
            else:
                bin_edges = np.append(bin_edges, edges, axis=0)

    n_bins_per_segment = len(bin_edges)/len(signal_slice_sets)
    segment_midpoints = np.array(segment_midpoints)

    parameters = Parameters()
    parameters['class'] = 0
    parameters['bin_edges'] = bin_edges
    parameters['n_segments'] = len(signal_slice_sets)
    parameters['n_bins_per_segment'] = n_bins_per_segment
    parameters['segment_midpoints'] = segment_midpoints
    parameters['location'] = location
    parameters['beta'] = beta

    return parameters


def read_signal(signal_x, signal_y, signal_slice_sets, axis):
    n_slices_per_bunch = signal_slice_sets[0].n_slices
    total_length = len(signal_slice_sets) * n_slices_per_bunch

    if (signal_x is None) or (len(signal_x) != total_length):
        signal_x = np.zeros(len(signal_slice_sets) * n_slices_per_bunch)

    if (signal_y is None) or (len(signal_x) != total_length):
        signal_y = np.zeros(len(signal_slice_sets) * n_slices_per_bunch)

    for idx, slice_set in enumerate(signal_slice_sets):
        idx_from = idx * n_slices_per_bunch
        idx_to = (idx + 1) * n_slices_per_bunch

        if axis == 'divergence':
            np.copyto(signal_x[idx_from:idx_to], slice_set.mean_xp)
            np.copyto(signal_y[idx_from:idx_to], slice_set.mean_yp)
        elif axis == 'displacement':
            np.copyto(signal_x[idx_from:idx_to], slice_set.mean_x)
            np.copyto(signal_y[idx_from:idx_to], slice_set.mean_y)
        else:
            return ValueError('Unknown axis')

    return signal_x, signal_y


def kick_bunches(local_slice_sets, bunch_list, local_bunch_indexes,
                 signal_x, signal_y, axis):

    n_slices_per_bunch = local_slice_sets[0].n_slices

    for slice_set, bunch_idx, bunch in zip(local_slice_sets,
                                           local_bunch_indexes, bunch_list):

        # the slice set data from all bunches in all processors pass the signal processors. Here, the correction
        # signals for the bunches tracked in this processors are picked by using indexes found from
        # mpi_gatherer.total_data.local_data_locations
        idx_from = bunch_idx * n_slices_per_bunch
        idx_to = (bunch_idx + 1) * n_slices_per_bunch

        # mpi_gatherer has also slice set list, which can be used for applying the kicks
        p_idx = slice_set.particles_within_cuts
        s_idx = slice_set.slice_index_of_particle.take(p_idx)

        if axis == 'divergence':
            if signal_x is not None:
                correction_x = np.array(signal_x[idx_from:idx_to], copy=False)
                bunch.xp[p_idx] -= correction_x[s_idx]
            if signal_y is not None:
                correction_y = np.array(signal_y[idx_from:idx_to], copy=False)
                bunch.yp[p_idx] -= correction_y[s_idx]

        elif axis == 'displacement':
            if signal_x is not None:
                correction_x = np.array(signal_x[idx_from:idx_to], copy=False)
                bunch.x[p_idx] -= correction_x[s_idx]
            if signal_y is not None:
                correction_y = np.array(signal_y[idx_from:idx_to], copy=False)
                bunch.y[p_idx] -= correction_y[s_idx]


class OneboxFeedback(object):
    def __init__(self, gain, slicer, processors_x, processors_y,
                 axis='divergence', mpi=False):

        if isinstance(gain, collections.Container):
            self._gain_x = gain[0]
            self._gain_y = gain[1]
        else:
            self._gain_x = gain
            self._gain_y = gain

        self._slicer = slicer

        self._processors_x = processors_x
        self._processors_y = processors_y


        self._axis = axis
        if axis == 'divergence':
            self._required_variables = ['mean_xp', 'mean_yp']
        elif axis == 'displacement':
            self._required_variables = ['mean_x', 'mean_y']

        self._required_variables = get_processor_variables(self._processors_x,
                                                     self._required_variables)
        self._required_variables = get_processor_variables(self._processors_y,
                                                     self._required_variables)
        # TODO: Normally n_macroparticles_per_slice is removed from
        #       the statistical variables. Check if it is not necessary.

        self._parameters_x = None
        self._parameters_y = None
        self._signal_x = None
        self._signal_y = None

        self._mpi = mpi
        if self._mpi:
            self._mpi_gatherer = mpi_data.MpiGatherer(self._slicer,
                                                      self._required_variables)
            self._local_bunch_indexes = self._mpi_gatherer.local_bunch_indexes
        else:
            self._local_bunch_indexes = [0]

    def track(self, bunch):
        if self._mpi:
            signal_slice_sets, bunch_slice_sets, bunch_list \
            = get_local_slice_sets(bunch, self._slicer, self._required_variables)
        else:
            signal_slice_sets, bunch_slice_sets, bunch_list \
            = get_mpi_slice_sets(bunch, self._mpi_gatherer)

        if self._parameters_x is None:
            self._parameters_x = generate_parameters(signal_slice_sets)
        if self._parameters_y is None:
            self._parameters_y = generate_parameters(signal_slice_sets)

        read_signal(self._signal_x, self._signal_y, signal_slice_sets,
                    self._axis)
        if self._signal_x is not None:
            kick_parameters_x, kick_signal_x = process(self._parameters_x,
                                                       self._signal_x,
                                                       self._processors_x,
                                                       slice_sets=signal_slice_sets)
            kick_signal_x = kick_signal_x * self._gain_x

        if self._signal_y is not None:
            kick_parameters_y, kick_signal_y = process(self._parameters_y,
                                                       self._signal_y,
                                                       self._processors_y,
                                                       slice_sets=signal_slice_sets)
            kick_signal_y = kick_signal_y * self._gain_y

        kick_bunches(bunch_slice_sets, bunch_list, self._local_bunch_indexes,
                 kick_signal_x, kick_signal_y, self._axis)

        if self._mpi:
            self._mpi_gatherer.rebunch(bunch)

class PickUp(object):
    def __init__(self, slicer, processors_x, processors_y, location_x, beta_x,
                 location_y, beta_y, mpi=False):

        self._slicer = slicer

        self._processors_x = processors_x
        self._processors_y = processors_y

        self._required_variables = ['mean_x', 'mean_y']

        self._required_variables = get_processor_variables(self._processors_x,
                                                     self._required_variables)
        self._required_variables = get_processor_variables(self._processors_y,
                                                     self._required_variables)
        # TODO: Normally n_macroparticles_per_slice is removed from
        #       the statistical variables. Check if it is not necessary.

        self._location_x = location_x
        self._beta_x = beta_x
        self._location_y = location_y
        self._beta_y = beta_y

        self._parameters_x = None
        self._parameters_y = None
        self._signal_x = None
        self._signal_y = None

        self._mpi = mpi
        if self._mpi:
            self._mpi_gatherer = mpi_data.MpiGatherer(self._slicer,
                                                      self._required_variables)
            self._local_bunch_indexes = self._mpi_gatherer.local_bunch_indexes
        else:
            self._local_bunch_indexes = [0]

    def track(self, bunch):
        if self._mpi:
            signal_slice_sets, bunch_slice_sets, bunch_list \
            = get_local_slice_sets(bunch, self._slicer, self._required_variables)
        else:
            signal_slice_sets, bunch_slice_sets, bunch_list \
            = get_mpi_slice_sets(bunch, self._mpi_gatherer)

        if self._parameters_x is None:
            self._parameters_x = generate_parameters(signal_slice_sets)
        if self._parameters_y is None:
            self._parameters_y = generate_parameters(signal_slice_sets)

        read_signal(self._signal_x, self._signal_y, signal_slice_sets,
                    self._axis)

        if self._signal_x is not None:
            end_parameters_x, end_signal_x = process(self._parameters_x,
                                                       self._signal_x,
                                                       self._processors_x,
                                                       slice_sets=signal_slice_sets)

        if self._signal_y is not None:
            end_parameters_y, end_signal_y = process(self._parameters_y,
                                                       self._signal_y,
                                                       self._processors_y,
                                                       slice_sets=signal_slice_sets)


class Kicker(object):
    def __init__(self, gain, slicer, processors_x, processors_y,
                 registers_x, registers_y, location_x, beta_x,
                 location_y, beta_y, combiner_type='vector_sum', mpi=False):

        if isinstance(gain, collections.Container):
            self._gain_x = gain[0]
            self._gain_y = gain[1]
        else:
            self._gain_x = gain
            self._gain_y = gain

        self._slicer = slicer

        self._processors_x = processors_x
        self._processors_y = processors_y

        self._registers_x = registers_x
        self._registers_y = registers_y

        if isinstance(combiner_type, str):
            if combiner_type == 'vector_sum':
                self._combiner_x = VectorSumCombiner(registers_x, location_x,
                                                   beta_x, np.pi/2.)
                self._combiner_y = VectorSumCombiner(registers_y, location_y,
                                                   beta_y, np.pi/2.)

            elif self._combiner_type == 'cosine_sum':
                self._combiner_x = CosineSumCombiner(registers_x, location_x,
                                                   beta_x, np.pi/2.)
                self._combiner_y = CosineSumCombiner(registers_y, location_y,
                                                   beta_y, np.pi/2.)

            elif self._combiner_type == 'hilbert':
                self._combiner_x = HilbertCombiner(registers_x, location_x,
                                                   beta_x, np.pi/2.)
                self._combiner_y = HilbertCombiner(registers_y, location_y,
                                                   beta_y, np.pi/2.)
            else:
                raise ValueError('Unknown combiner type')
        else:
            self._combiner_x = self._combiner_type(registers_x, location_x,
                                                   beta_x, np.pi/2.)
            self._combiner_y = self._combiner_type(registers_y, location_y,
                                                   beta_y, np.pi/2.)

        self._required_variables = ['mean_xp', 'mean_yp']
        self._required_variables = get_processor_variables(self._processors_x,
                                                     self._required_variables)
        self._required_variables = get_processor_variables(self._processors_y,
                                                     self._required_variables)
        # TODO: Normally n_macroparticles_per_slice is removed from
        #       the statistical variables. Check if it is not necessary.

        self._mpi = mpi
        if self._mpi:
            self._mpi_gatherer = mpi_data.MpiGatherer(self._slicer,
                                                      self._required_variables)
            self._local_bunch_indexes = self._mpi_gatherer.local_bunch_indexes
        else:
            self._local_bunch_indexes = [0]

    def track(self, bunch):
        if self._mpi:
            signal_slice_sets, bunch_slice_sets, bunch_list \
            = get_local_slice_sets(bunch, self._slicer, self._required_variables)
        else:
            signal_slice_sets, bunch_slice_sets, bunch_list \
            = get_mpi_slice_sets(bunch, self._mpi_gatherer)

        if (self._combiner_x is None) or (self._combiner_y is None):
            self.__init_combiners

        parameters_x, signal_x = self._combiner_x.process()
        parameters_y, signal_y = self._combiner_y.process()

        if self._signal_x is not None:
            parameters_x, signal_x = process(parameters_x, signal_x,
                                             self._processors_x,
                                             slice_sets=signal_slice_sets)
            signal_x = signal_x * self._gain_x

        if self._signal_y is not None:
            parameters_y, signal_y = process(parameters_y, signal_y,
                                             self._processors_y,
                                             slice_sets=signal_slice_sets)
            signal_y = signal_y * self._gain_y

        kick_bunches(bunch_slice_sets, bunch_list, self._local_bunch_indexes,
                     signal_x, signal_y, self._axis)

        if self._mpi:
            self._mpi_gatherer.rebunch(bunch)
