# Signal processor specifications

This file contains specifications for signal processors, which can be used in the feedback module of PyHEADTAIL or as a separate tool for studying feedback systems from the control theory point of view.

The concept of signal processor might be rather abstract, and the purpose of this document is not to scare users. Thus, at first, it is recommended to explore examples in the examples folder. After that, if one wants to develope a new signal processor, it is recommended to copy the code of *MinimalSignalProcessor* from below and play with that. Only after that, when problems occur or, especially, when the new signal processor is ready for other users, it is recommended to read this document carefully.

### Definition of signal
Before specifications for signal processors can be discussed, the concept of signal must be defined. It might sound simple, but is it? Someone could define that a signal is time varying, finite or infinite length voltage or current signal, i.e. an analog signal. For someone else it might be a list of numbers where each number represents a value at a certain moment of equally spaced time, i.e. a digital signal. There are also situations, where signals are much nastier, e.g. not equally spaced nor even continuous in time.

Unfortunately, the required definition for signals should work in the all cases discussed above. Because it is very unpractical if not almost impossible to develop all signal processors to work with the all types of signals, three definition for signals are introduced basing on the assumptions required by the definition. Thus, signal processors can be specified to receive and transmit signals belonging to these classes.

There is a hierarchy in the signal classes. For class 0 signals a very few assumptions about the signal has been made. Class 1 signals are assumed to be equally spaced, but not continuous in time. In Class 2, signals are finite length, equally spaced and continuous in time. This means that a signal processor designed for lower class signals (e.g. Class 0) can process also signals from higher classes (e.g. Class 1 and Class 2), but not vice versa.

The basic definition for a signal is that it is a numpy array of floating point numbers. Each number in the array corresponds to an averaged signal over the specific time interval, *bin*. There are no limitations how the bins are located in the physical space. Thus, additional data are given together with the signal in the variable *signal_parameters*. The namedtuple *signal_parameters* can be created by importing it from the file *signal.py*, i.e.

~~~python
from signal import SignalParameters

signal_parameters = SignalParameters(signal_class, bin_edges, n_segments, n_bins_per_segment,
                                                                original_segment_mids, phase_advance])
~~~

It includes six auxiliary parameters for the signal:

    - *signal class*: a signal class
    - *bin_edges*: a 2D numpy array, which is equal length to the signal, but each row includes two floating point numbers, the positions of the bin edges in the physical space
    - *n_segments*: a number of equal length and equally binned segments where to the signal can be divided
    - *n_bins_per_segment*: a number of bins per segment.
    `len(bin_edges)/n_segments`
    - *original_segment_mids*: a numpy array of original middle points for the segments
    - *phase_advance*: a location of the signal in betatron phase

Note, that it is not allowed to modify parameters *n_segments* or *original_mids* in the signal processors. This is because it might be necessary to know the original middle points of the segments/bunches in some signal processors as reference points

##### Class 0
There are no limitations for Class 0 signal, i.e. bin spacing and bin length might vary randomly. If the signal can be divided into segments, each segment must have an equal number of bins and bin spacing and bin lengths must be equal for each segment.

Class 0 signal gives a large freedom to use any kind of signal as an input for the signal processors. Particularly it means that a single array of the slice values from multiple bunches can be used directly as a signal.

##### Class 1
In this class, it is assumed that signal might be divided into equal length sequences which are separated by empty spaces. Bin spacing and width must be constant and equal in each segment.

In practice this means that signals from each bunch has an equal number of equally spaced slices/samples.

##### Class 2
Signal is equally spaced and continuous in time.

In practice this means that the signal is continuously sliced/sampled over all bunches including empty spaces between bunches. This also limits the slicing/sampling rate to be a fraction of the bunch spacing in the case of multi bunch simulations.


### Signal processors
A signal processor is a Python object which processes/modifies signals. The signal processing occurs in the method *process(signal_parameters, signal, ...)*, which takes arguments *signal_parameters* and *signal* and returns possibly modified versions of them. The code of the minimal signal processor is following:
~~~python
class MinimalSignalProcessor(object)
    def __init__(self):
        self.signal_classes = (0,0)
        self.extensions = []

    def process(self, signal_parameters, signal, *args, **kwargs):

        # the signal or the signal_parameters could be modified here

        return signal_parameters, signal
~~~
Additional argument for the method *process(...)* are allowed and, therefore, **args* and ***kwargs* must be included to the list of input arguments. The allowed classes for incoming and outgoing signals must be specified in the tuple *signal_classes*.

The (standardized) extensions for the minimal processor layout can be applied by adding the names of the extensions to the list *extensions*. Those extensions might allow additional data for signal manipulations, assistant for debugging/visualization, etc, and the processor compatibility can be easily checked by checking the names from the list *extensions*.

##### Extension: bunch
This extension provides additional information from the simulated bunche(s) to the processors by guaranteeing that a list of PyHEADTAIL slice set objets for the bunches is an argument for the method *process*. In order to limit a number of statistical parameters calculated for the slice sets, the required variables (*n_macroparticles_per_slice*, *mean_x*, *mean_y*, *mean_z*, *mean_xp*, *mean_yp*, *mean_dp*,*sigma_x*, *sigma_y*, *sigma_z*, *sigma_dp*, *epsn_x*, *epsn_y* and/or *epsn_z*) must be listed in the variable *required_variables*


~~~python
class BunchExtendedProcessor(object)
    def __init__(self):
        self.signal_classes = (0,0)
        self.extensions = ['bunch']

        self.required_variables = []

    def process(self, signal_parameters, signal, slice_sets,*args, **kwargs):

        # the signal or the bin_edges could be modified here

        return signal_parameters, signal
~~~


##### Extension: store
In order to help debugging and data visualization, the incoming and outgoing signal parameters and signals can be stored into the processor by using extension *store*. The extension can be activated by setting the input parameter *store* to *True*. In that case, the incoming and outgoing signal_parameters and signals are stored to the variables *input_signal*, *input_signal_parameters*, *output_signal* and *output_signal_parameters*.

~~~python
class StoreExtendedProcessor(object)

    def __init__(self, store = False):

        self.label = 'Bypass'
        self._store_signal = store_signal

        self.input_signal = None
        self.input_signal_parameters = None

        self.output_signal = None
        self.output_signal_parameters = None

    def process(self, signal_parameters, signal, *args, **kwargs):

        # signal could be modified here
        output_signal_parameters = copy.copy(signal_parameters)
        output_signal = np.copy(signal)

        if self._store_signal:
            self.input_signal = np.copy(signal)
            self.input_signal_parameters = copy.copy(signal_parameters)
            self.output_signal = np.copy(output_signal)
            self.output_signal_parameters = copy.copy(output_signal_parameters)

        return output_signal_parameters, output_signal
~~~

##### Extension: register
A signal generator can be also used to store. For that purpose, the register extension has been developed. The register extension has two different purposes: it stores the data and it provides a tool to rotate the data to the reader betatron phase angle. Because the betatron phase rotation can be done multiple different ways, the extension is rather abstract:

    - the processor is iterable, i.e. it contains methods *__\_\_iter\_\_()__* and *__next()__*. The iterator returns the signals from different turns.
    - the processor contains a function namely *__combine(...)__*, which returns a signal turned into the reader betatron phase. The method takes two signal from different betatron phases as input arguments.
    - the processor has a variable namely *__combination__* which is a string containing 'combined' if the method combine requires two signal or 'individual' if it can do the rotation using only a single value.

Note that the specifications for the register extencion will be probably changed/simplified in the near future.

## Types of signal processors

In order to simplify development and maintenance of the code, the most of the signal processors are based on abstract classes. At the moment there are six abstract classes basing on the basic mathematical methods

#### Addition
Calculates a sum of the signal and an equally length NumPy array. Can handle signals belonging to the class 0.

#### Multiplication
Multiplies the signal and an equally length NumPy array element by element. Can handle signals belonging to the class 0.

#### Linear transform
Multiplies a square matrix with the signal. The multiplication can be done by segment by segment or all at once. Note that in the latter case the size of matrix might be huge and its creation might take a very long time. Can handle signals belonging to the class 0.

#### Convolution
Calculates a convolution between the signal and an impulse response. The multiplication can be done by segment by segment or all at once.

Note that many processors can be implemented both by using the convolution or the linear transform. The main difference is that the convolution is significantly faster (order(s) of magnitude on the first turn) but requires a class 1 signal, whereas the linear transform can handle class 0 signals

#### Resampling
Changes the bin set of the signal. Also the class of the signal can be changed (i.e 0 -> 1/2 or vice versa). The operation is done by segment by segment.

#### Register
Can store signals over mutliple turns and turn them into the kicker betatron phase.

## Open questions / TODO?

* The unit of the bin edges from meter to second?