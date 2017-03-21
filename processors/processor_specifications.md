# Signal processor specifications

This file contains specifications for signal processors, which can be used in the framework for finite length signal processing (e.g. the transverse feedback module for PyHEADTAIL).

The concept of signal processor might be rather abstract, and the purpose of this document is not to scare the users. Thus, at first, it is recommended to explore examples. After that, if one wants to develope a new signal processor, it is recommended to copy the code of *MinimalSignalProcessor* from below and play with that. Only after that, when problems occur or, especially, when the new signal processor is ready for other users, it is recommended to read this document carefully.

### Definition of signal
Before specifications for signal processors can be discussed, the concept of signal must be defined. It might sound simple, but is it? Someone could define that a signal is time varying, continuous, finite or infinite length, quantity, i.e. an analog signal. For someone else it might be a list of numbers where each number represents a value at a certain moment of equally spaced time, i.e. a digital signal. There are also situations, where the signals are much nastier, e.g. not equally spaced nor even continuous in time.

Unfortunately, the required definition of signal should work in all of the cases mentioned above. Because it is very unpractical if not almost impossible to develop all signal processors to work with the all types of signals, the definition of the signal is split into three levels basing on the assumptions which can be made about the signal. Thus, signal processors can be specified to receive and transmit signals in the specific classes. Due to the hierarchy of the signal classes, a signal processor designed for lower class signals is also able to process signals from higher classes.

The basic definition for a signal is that it is a numpy array of floating point numbers. Each number in the array corresponds to an averaged signal over the specific time interval, *bin*. There are no limitations how the bins are located in the physical space for the lowest level signals. Thus, additional data are given together with the signal in the variable *parameters*. It is a Python dictionary which contains at least the following elements:

    - *class*: a signal class
    - *bin_edges*: a 2D numpy array, which is equal length to the signal. Each row includes two floating point numbers, the edge positions of the bin in the physical space
    - *n_segments*: a number of equal length and equally binned segments where to the signal can be divided
    - *n_bins_per_segment*: a number of bins per segment. `len(bin_edges)/n_segments`
    - *segment_midpoints*:  a numpy array of original midpoints of the segments
    - *location*: a location of the signal in betatron phase.
    - *beta*: a vale of beta function in the source of the signal. Value 1 is neutral for signal processing

All the other parameters than *n_segments* and *segment_midpoints* can be modified in signal processors. Those two parameters are required to stay unchanged in order to keep the signal synchronized with the original bin set during the resampling and possibly parallel computing. Thus, the signal can be resampled back to the original bin set without extra user parameters.

##### Class 0
There are no limitations for Class 0 signals, i.e. bin spacing and bin length might vary randomly. If the signal can be divided into segments, each segment must have an equal number of bins with equal bin spacing and bin lengths.

Class 0 signal gives a large freedom to use any kind of signal as an input for the signal processors. Particularly it means that a single array of the slice values from multiple bunches can be used directly as a signal.

##### Class 1
In this class, it is assumed that signal might be divided into equal length sequences which are separated by empty spaces. Bin spacing and width must be constant and equal in each segment.

In practice this means that signals from each bunch has an equal number of equally spaced slices/samples.

##### Class 2
Signal is equally spaced and continuous in time.

In practice this means that the signal is continuously sliced/sampled over all bunches including empty spaces between bunches. This also limits the slicing/sampling rate to be a fraction of the bunch spacing in the case of multi bunch simulations.


### Signal processors
A signal processor is a Python object which processes/modifies signals. The signal processing occurs in the method *process(parameters, signal, ...)*, which takes arguments *parameters* and *signal* and returns possibly modified versions of them. The example code for a minimal signal processor is following:
~~~python
class MinimalSignalProcessor(object)
    def __init__(self):
        self.signal_classes = (0,0)
        self.extensions = []

    def process(self, parameters, signal, *args, **kwargs):

        # the signal or the parameters could be modified here

        return parameters, signal
~~~
Additional argument for the method *process(...)* are allowed and, therefore, **args* and ***kwargs* must be included to the list of input arguments. The allowed classes for incoming and outgoing signals must be specified in the tuple *signal_classes*.

The (standardized) extensions for the minimal processor layout can be applied by adding the names of the extensions to the list *extensions*. Those extensions might allow additional data for signal manipulations, assist debugging/visualization, etc. Compatibility of the processors can be easily checked and scripted by checking the names from the list *extensions*.

##### Extension: bunch
This extension provides additional information from the simulated bunche(s) to the processors by guaranteeing that a list of PyHEADTAIL slice set objets from the bunches is an argument for the method *process*. In order to limit a number of statistical parameters calculated for the slice sets, the required variables (*n_macroparticles_per_slice*, *mean_x*, *mean_y*, *mean_z*, *mean_xp*, *mean_yp*, *mean_dp*,*sigma_x*, *sigma_y*, *sigma_z*, *sigma_dp*, *epsn_x*, *epsn_y* and/or *epsn_z*) must be listed in the list *required_variables*


~~~python
class BunchExtendedProcessor(object)
    def __init__(self):
        self.signal_classes = (0,0)
        self.extensions = ['bunch']

        self.required_variables = []

    def process(self, parameters, signal, slice_sets,*args, **kwargs):

        # the signal or the bin_edges could be modified here

        return parameters, signal
~~~


##### Extension: debug
In order to help debugging and data visualization, the incoming and outgoing signal parameters and signals can be stored into the processor by using extension *debug*. The extension can be activated by setting the input parameter *debug* to *True*. In that case, the incoming and outgoing parameters and signals are stored to the variables *input_signal*, *input_parameters*, *output_signal* and *output_parameters*.

~~~python
class DebugExtendedProcessor(object)

    def __init__(self, debug = False):

        self.label = 'Bypass'
        self._debug = debug

        self.input_signal = None
        self.input_parameters = None

        self.output_signal = None
        self.output_parameters = None

    def process(self, parameters, signal, *args, **kwargs):

        # signal could be modified here, e.g.
        ouput_parameters, output_signal = doSomething(parameters, signal)

        if self._debug:
            self.input_signal = np.copy(signal)
            self.input_parameters = copy.copy(signal_parameters)
            self.output_signal = np.copy(output_signal)
            self.output_parameters = copy.copy(output_parameters)

        return output_parameters, output_signal
~~~

##### Extension: register
A register is a special signal processor, which stores signals. Effectively it bypasses the signal unchanged.
A register is an iterable object, i.e. iteration returns the stored parameters, signal and and delay to the reading moment in phase advance. The standard properties for a register are given in the example code below.

~~~python
class Register(object):
    def __init__(self, n_values, tune, delay=0, **kwargs):


        self._n_values = n_values
        self._delay = delay
        self._phase_advance_per_turn = 2. * np.pi * tune

        self._n_iter_left = 0
        self._signal_register = deque(maxlen=(n_values + delay))
        self._parameter_register = deque(maxlen=(n_values + delay))

        self.extensions = ['register']

    @property
    def parameters(self):
        if len(self._parameter_register) > 0:
            return self._parameter_register[0]
        else:
            return None

    @property
    def phase_advance_per_turn(self):
        return self._phase_advance_per_turn

    @property
    def delay(self):
        return self._delay

    @property
    def maxlen(self):
        return self._n_values

    def __len__(self):
        return max((len(self._signal_register) - self._delay), 0)

    def __iter__(self):
        self._n_iter_left = len(self)
        return self

    def next(self):
        if self._n_iter_left < 1:
            raise StopIteration

        else:
            delay = -1. * (len(self._signal_register) - self._n_iter_left) \
                            * self._phase_advance_per_turn
            self._n_iter_left -= 1

            return (self._parameter_register[self._n_iter_left],
                    self._signal_register[self._n_iter_left], delay)

    def process(self, parameters, signal, *args, **kwargs):

        self._parameter_register.append(parameters)
        self._signal_register.append(signal)

        return parameters, signal
~~~


##### Extension: combiner
Combiners are objects which use registers as a signal source. A list of registers is given to Combiner as input argument. It is not quaranteed that a combiner is in the middle of a signal processor chain, i.e. that the process(...) method gets parameters and a signal as input arguments. Note that the order and the names of the input parameters for combiners (registers, target_location, target_beta, additional_phase_advance and beta_conversion) are standarized.

~~~python

class Combiner(object):

    def __init__(self, registers, target_location, target_beta=None,
                 additional_phase_advance=0., beta_conversion = '0_deg', **kwargs):

        self._registers = registers
        self._target_location = target_location
        self._target_beta = target_beta
        self._additional_phase_advance = additional_phase_advance
        self._beta_conversion = beta_conversion

        if self._beta_conversion == '0_deg':
            pass
        elif self._beta_conversion == '90_deg':
            self._additional_phase_advance += pi/2.
        else:
            raise ValueError('Unknown beta conversion type.')


        self.extensions = ['combiner']


    def process(self, *args, **kwargs):

        for register in registers:
            for parameters, signal, delay in register:
                output_parameters, output_signal = self.DoSomething(parameters, signal, delay)

        return output_parameters, output_signal

~~~

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
Changes the bin set of the signal. Also the class of the signal can be changed (i.e 0 -> 1 or 2 or vice versa). The operation is done by segment by segment.

#### Register
Can store signals over mutliple turns. The signal can be rotated in betatron phase by using combiners which take references to registers as an input paramter.

## Open questions / TODO?

* The unit of the bin edges from meter to second?