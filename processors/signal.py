import collections

SignalParameters = collections.namedtuple('SignalParameters', ['signal_class','bin_edges','n_segments',
                                                               'n_slices_per_segment', 'phase_advance',
                                                               'original_z_mids'])