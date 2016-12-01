import numpy as np

class Bypass(object):
    """ A fast bypass processor, whichi does not modify the signal. A black sheep, which does not fit for
        the abstract classes.
    """

    def __init__(self, store_signal = False):
        self.label = 'Bypass'
        self.required_variables = []
        self._store_signal = store_signal

        self.input_signal = None
        self.input_bin_edges = None

        self.output_signal = None
        self.output_bin_edges = None

    def process(self,bin_edges, signal, slice_sets, phase_advance=None):
        if self._store_signal:
            self.input_signal = np.copy(signal)
            self.input_bin_edges = np.copy(bin_edges)
            self.output_signal = np.copy(signal)
            self.output_bin_edges = np.copy(bin_edges)

        return bin_edges, signal