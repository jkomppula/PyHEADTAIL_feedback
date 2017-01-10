# Signal processor specifications

This file contains specifications for a signal processor, which can be used in the feedback module of PyHEADTAIL or in a separate tool for studying feedback systems from the control theory point of view.

The concept of signal processor might be rather abstract, and the purpose of this document is not to scare users. Thus, at first, it is recommended to explore examples in the examples folder. After that, if one wants to develope a new signal processor, it is recommended to copy the code of *MinimalSignalProcessor* from below and play with that. Only after that, when problems occur or, especially, when the new signal processor is ready for other users, it is recommended to read this document carefully.

### Definition of signal
Before specifications for signal processors can be discussed, the concept of signal must be defined. It might sound simple, but is it? Someone could define that a signal is time varying, finite or infinite length voltage or current signal, i.e. an analog signal. For someone else it might be a list of numbers where each number represents a value at a certain moment of equally spaced time, i.e. a digital signal. There are also situations, where signals are much nastier, e.g. not equally spaced nor even continuous in time.

Unfortunately, the required definition for signals should work in the all cases discussed above. Because it is very unpractical if not almost impossible to develop all signal processors to work with the all types of signals, three definition for signals are introduced basing on the assumptions required by the definition. Thus, signal processors can be specified to receive and transmit signals belonging to these classes.

There is a hierarchy in the signal classes. For class 0 signals a very few assumptions about the signal has been made. Class 1 signals are assumed to be equally spaced, but not continuous in time. In Class 2, signals are finite length, equally spaced and continuous in time. This means that a signal processor designed for lower class signals (e.g. Class 0) can process upper class signals  (e.g. Class 1 and Class 2), but not vice versa.

~~~python
import collections

SignalParameters = collections.namedtuple('SignalParameters', ['class','bin_edges','n_segments', 'n_slices_per_segment'])
~~~


##### Class 0
The basic definition for a signal is that it is a numpy array of floating point numbers. Each number in the array corresponds to an averaged signal over the specific time interval, *bin*. There are no limitations how the bins are located in the physical space. Thus, additional data are given together with the signal in the variable *bin_edges*. A 2D numpy array *bin_edges* is equal length to the signal, but each row includes two floating point numbers, the positions of the bin edges in the physical space.

The definition above might be complicated and abstract, but there is a reason for that. In macroparticle simulation codes the bunch is divided into slices. For the reasons of simulation techniques (e.g. numerical stability and numerical noise), it is beneficial, in some cases, that the slice width is not constant over the bunch. Furthermore, even if the slice width is constant, because of the large distance/time scale differences between an accelerator (~km), a bunch (~m) and a slice (~cm), it is numerically challenging to fill entire accelerator with equally spaced slices, i.e. the signal is not continuous.

Class 0 signal gives a large freedom to use any kind of signal as an input for the signal processors valid for this class. Particularly it means that a single array of the slice values from multiple bunches can be used directly as a signal.

##### Class 1
In this class, it is assumed that the bin width is constant but the signal might be divided into sequences by empty time intervals which are not covered by bins.
In order to simplify development of signal processors it is also assumed that there is an equal number of bins in each sequence and the number of sequences is invariant. All together, the assumptions are:

    * The bin width is constant over the signal.
    * The signal might be divided into sequences
        * Sequences can be separated by empty time intervals which are not covered by bins.
        * The lengths of the empty time intervals might vary
        * There are an equal number of bins in each sequence
        * The number of sequences is not changed in any case in any processor

In practice this means that signals from each bunch has an equal number of equally spaced slices/samples.

##### Class 2
Signal is equally spaced and continuous in time. In the other words, the assumptions are:

    * The bin width is constant over the signal.
    * There are no empty time intervals in the signal
    * In order to guarantee compatibility with Class 1 signals, the number of bins must be divisible by the number of sequences, if the signal could be divided into sequences at some point of the signal processing

In practice this means that the signal is continuously sliced/sampled over all bunches including empty spaces between bunches. In many cases this means also that the slicing/sampling rate must be a fraction of the bunch spacing

### Signal processors
A signal processor is a Python object which processes/modifies signals. The signal processing occurs in the method *process(signal_parameters, signal, ...)*, which takes arguments *signal_parameters* and *signal* and returns possibly modified versions of them. The code of the minimal signal processor is following:
~~~python
class MinimalSignalProcessor(object)
    def __init__(self):
        self.signal_classes = (0,0)
        self.extensions = []

    def process(self, signal_parameters, signal, *args, **kwargs):

        # the signal or the bin edges could be modified here

        return signal_parameters, signal
~~~
Additional argument for the method *process(...)* are allowed and, therefore, **args* and ***kwargs* must be included to the list of input arguments. The variable *signal_parameters* is a tuple which contains a list of bin_edges and a number of sequences in the signal, i.e. *signal_parameters = (bin_edges,n_sequences)*. The allowed classes for incoming and outgoing signals must be specified in the tuple *signal_classes*.

The standard for signal processors can be extended from the minimal, for example, in order to provide additional data for signal manipulations or assist debugging and visualization. The extensions supported by the processor must be listed to the list *extensions*, which is empty by default.

##### Extension: bunch
This extension provides additional information from the simulated bunche(s) to the processor. This has been done by adding two extra arguments to the method *process(...)*; *slice_sets* and *data_location*. Slice_sets is a list of PyHEADTAIL slice set objects from all the simulated bunches. *Data location* is a tuple, which includes values of betatron phase advance and beta function in the location where the signal origins.

The possible variables are *mean_x*, *mean_y*, *sigma_x*, *sigma_y*

Note also that it is not quaranteet that


~~~python
class MinimalSignalProcessor(object)
    def __init__(self):
        self.signal_classes = (0,0)
        self.extensions = ['bunch']

        self.required_variables = []

    def process(self, signal_parameters, signal, slice_sets, data_location, *args, **kwargs):
        print 'I have data from ' + str(len(slice_sets)) + 'bunches!'
        print ''

        print 'That data is from the location where:'
        print 'The betatron phase advance is ' + str(data_location[0]) + ' rad'
        print 'Betafunction is ' + str(data_location[1]) + ' m'



        # the signal or the bin_edges could be modified here

        return , signal_parameters, signal
~~~


##### Extension: store


~~~python
class Bypass(object):

    def __init__(self, store_signal = False):


        self.label = 'Bypass'
        self._store_signal = store_signal

        self.input_signal = None
        self.input_bin_edges = None

        self.output_signal = None
        self.output_bin_edges = None

    def process(self, signal_parameters, signal, *args, **kwargs):

        # signal could be modified here

        if self._store_signal:
            self.input_signal = np.copy(signal)
            self.input_bin_edges = np.copy(bin_edges)
            self.output_signal = np.copy(signal)
            self.output_bin_edges = np.copy(bin_edges)

        return , signal_parameters, signal
~~~

##### Extension: register


## Types of signal processors

In order to simplify development and maintenance of the code, the most of the signal processors are based on abstract classes.
### Addition

### Multiplication

### Convolution

### Linear transform

### Register

    A general requirement for the signal processor is that it is a class object containing a function, namely,
    process (signal, slice_set, phase_advance). The input parameters for the function process(signal, slice_set) are
    a numpy array 'signal',  a slice_set object of PyHEADTAIL and a phase advance of the signal in the units of absolute
    angle of betatron motion from the reference point of the accelerator. The function must return a numpy array with
    equal length to the input array. The other requirement is that the class object contains a list variable, namely
    'required_variables', which includes required variables for slicet_objects.

    The signals processors in this file are based on four abstract classes;
        1) in LinearTransform objects the input signal is multiplied with a matrix.
        2) in Multiplication objects the input signal is multiplied with an array with equal length to the input array
        3) in Addition objects to the input signal is added an array with equal length to the input array
        4) A normal signal processor doesn't store a signal (in terms of process() calls). Processors buffering,
           registering and/or delaying signals are namely Registers. The Registers have following properties in addition
           to the normal processor:
            a) the object is iterable
            b) the object contains a function namely combine(*args), which combines two signals returned by iteration
               together


## Open questions / TODO?

* The unit of the bin edges from meter to second?
* The number of sequences as a required specification for signals
*