import collections
""" This file contains the core code for the feedback simulations including basic data stuctures for signals as well as
    functions for passing signal through the signal processors, etc. The code can be used for building interfaces between
    the framework and PyHEADTAIL or other simplified models for bunches.
"""



""" External parameters describing signal:
        signal_class: A class of the signal, which determines what kind of assumptions can be made from the signal.
                See more details from the file processor_specifications.md
        bin_edges: A 2D numpy array, where each row determines edges of the bin (value of the signal)
        n_segments: A number of equally length segments with equal bin pattern the signal can be divided
        n_bins_per_segment: A number of bins in the segments determined above
        original_segment_mids: This value is used as a reference point in some signal processors (i.e. resampling)
        beam_parameters: Beam parameters which determined by a physical location in the accelerator. The input is
                a namedtuple containing a betatron phase advance and a value of the betafunction (see below).
"""
SignalParameters = collections.namedtuple('SignalParameters', ['signal_class','bin_edges','n_segments',
                                                               'n_bins_per_segment',
                                                               'original_segment_mids', 'beam_parameters'])

""" A namedtuple which contains beam parameters related to a physical location in the accelerator, i.e. a location in
    the betatron phase advance from the reference point of the accelerator and a value of the beta function

"""
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


def process(signal_parameters,signal, processors ):
    pass