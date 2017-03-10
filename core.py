import collections
import numpy as np

""" This file contains the core functions and variables for signal processing.
    That code can be used for implementing interfaces for other codes and
    libraries (e.g. PyHEADTAIL).
"""

""" External parameters describing signal:
        signal_class: A class of the signal, which determines what kind of
            assumptions can be made from the signal. See more details from
            the file processor_specifications.md
        bin_edges: A 2D numpy array, where each row determines edges of
            the bin (value of the signal)
        n_segments: A number of equally length segments with equal bin pattern
            the signal can be divided
        n_bins_per_segment: A number of bins in the segments determined above
        original_segment_mids: This value is used as a reference point in some
            signal processors (i.e. resampling)
        additional_parameters: A dictionary of additional paramters which are
            carried with the signal (e.g. beam parameteres).
"""


def Parameters():
    """ Returns a prototype for signal parameters."""
    return {'class': 0, 'bin_edges': np.array([]), 'n_segments': 0,
              'n_bins_per_segment': 0, 'segment_midpoints': np.array([])}


def Signal():
    """ Returns a prototype for a signal."""
    return np.array([])


def process(parameters, signal, processors, **kwargs):
    """
    A function which processes the signal, i.e. passes the signal through the signal processors
    :param signal_parameters: A standardized namedtuple for additional parameters for the signal
    :param signal: A Numpy array, which is the actual signal to be processed
    :param processors: A list of signal processors
    :param **kwargs: Extra parameters related to the extensions of the processors (e.g. slice_set)
    :return:
    """

    for processor in processors:
        parameters, signal = processor.process(parameters,
                                                      signal, **kwargs)

    return parameters, signal


def get_processor_extensions(processors, available_extensions=None):
    """ A function, which checks available extensions from the processors
    """

    if available_extensions is None:
        available_extensions = []

    for processor in processors:
        if processor.extensions is not None:
            available_extensions.extend(processor.extensions)

    available_extensions = list(set(available_extensions))

    return available_extensions



# Extension specific code
#########################


""" A namedtuple which contains beam parameters related to a physical location in the accelerator, i.e. a location in
    the betatron phase advance from the reference point of the accelerator and a value of the beta function

"""
# TODO: rename beta_function to beta
BeamParameters = collections.namedtuple('BeamParameters', ['phase_advance','beta_function'])

def get_processor_variables(processors, required_variables = None):
    """Function which checks bunch variables required by signal processors. In PyHEADTAIL bunch variables are
        the statistical variables of a slice_set object inlcuding n_macroparticles_per_slice. See more details from
        the document processors/processor_specifications.md.

    :param processors: a list of signal processors
    :param required_variables: an additional list of bunch variables
    :return: a list of bunch variables, which is a combination of those variables given as input parameter and
        found from the processors
    """

    if required_variables is None:
        required_variables = []

    for processor in processors:
        if 'bunch' in processor.extensions:
            required_variables.extend(processor.required_variables)

    required_variables = list(set(required_variables))

    if 'z_bins' in required_variables:
        required_variables.remove('z_bins')

    return required_variables


#TODO:
def combine(target_beam_parameters,registers):
    """This will be a general function for combining signal from different betatron phase advances """
    target_phase_advance = target_beam_parameters.phase_advance
    target_beta = target_beam_parameters.beta_function

    source_phase_advances = []
    source_betas = []

    for register in registers:
        source_phase_advances.append(register.beam_parameters.phase_advance)
        source_betas.append(register.beam_parameters.beta_function)

    target_signal = None
    n_source_signal = None



    total_signal = None
    n_signals = 0

    if len(registers) == 1:
        # If there is only one register, uses signals from different turns for combination

        prev_signal = None
        for signal in registers[0]:
            if total_signal is None:
                prev_signal = signal
                total_signal = np.zeros(len(signal[0]))
            phase_conv_coeff = 1. / np.sqrt(beam_parameters.beta_function * registers[0].beam_parameters.beta_function)
            total_signal += phase_conv_coeff * registers[0].combine(signal, prev_signal, target_phase_advance, True)
            n_signals += 1
            prev_signal = signal

    else:
        # if len(registers) == 2 and registers[0].combination == 'combined':

        if registers[0].combination == 'combined':
            # If there are only two register and the combination requires signals from two register, there is only
            # one pair of registers
            prev_register = registers[0]
            first_iterable = 1
        else:
            # In other cases, loop can go through all successive register pairs
            prev_register = registers[-1]
            first_iterable = 0

        for register in registers[first_iterable:]:
            # print prev_register
            # print register
            phase_conv_coeff = 1. / np.sqrt(beam_parameters.beta_function * prev_register.beam_parameters.beta_function)
            for signal_1, signal_2 in itertools.izip(prev_register, register):
                if total_signal is None:
                    total_signal = np.zeros(len(signal_1[0]))
                total_signal += phase_conv_coeff * prev_register.combine(signal_1, signal_2, reader_phase_advance, True)
                n_signals += 1
            prev_register = register

    if total_signal is not None:
        total_signal /= float(n_signals)

    return total_signal

# general generator for signals?
