import collections
""" This file contains core data types and functions for the feedback simulations. These , i.e.


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



def process(signal_parameters,signal, processors ):
    pass