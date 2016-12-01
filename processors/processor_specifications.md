"""
    This file contains signal processors which can be used in the feedback module in PyHEADTAIL.

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

    @author Jani Komppula
    @date 16/09/2016
    @copyright CERN

"""
