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

    def process(self,bin_edges, signal, *args, **kwargs):
        if self._store_signal:
            self.input_signal = np.copy(signal)
            self.input_bin_edges = np.copy(bin_edges)
            self.output_signal = np.copy(signal)
            self.output_bin_edges = np.copy(bin_edges)

        return bin_edges, signal


class Average(object):

    def __init__(self, avg_type = 'bunch', store_signal = False):
        self.label = 'Average'
        self._avg_type = avg_type

        self.required_variables = []
        self._store_signal = store_signal

        self.input_signal = None
        self.input_bin_edges = None

        self.output_signal = None
        self.output_bin_edges = None

    def process(self, bin_edges, signal, slice_sets, *args, **kwargs):

        if self._avg_type == 'bunch':
            n_bunches = len(slice_sets)
            n_slices_per_bunch = len(signal) / n_bunches

            output_signal = np.zeros(len(signal))
            ones = np.ones(n_slices_per_bunch)

            for i in xrange(n_bunches):
                idx_from = i * n_slices_per_bunch
                idx_to = (i + 1) * n_slices_per_bunch
                np.copyto(output_signal[idx_from:idx_to], ones * np.mean(signal[idx_from:idx_to]))

        elif self._avg_type == 'total':
            output_signal = np.ones(len(signal))*np.mean(signal)

        else:
            raise ValueError('Unknown value in Average._avg_type')

        if self._store_signal:
            self.input_signal = np.copy(signal)
            self.input_bin_edges = np.copy(bin_edges)
            self.output_signal = np.copy(output_signal)
            self.output_bin_edges = np.copy(bin_edges)



        return bin_edges, output_signal