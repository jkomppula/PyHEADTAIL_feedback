from __future__ import division
import sys, os
BIN = os.path.expanduser("../../../../")
sys.path.append(BIN)
import sys
import copy
import datetime
import numpy as np
from scipy.constants import c, e
import matplotlib.pyplot as plt
from PyHEADTAIL_feedback.signal_tools.signal_generators import SimpleBeam, CircularPointBeam
from PyHEADTAIL_feedback.signal_tools.trackers_and_kickers import track_beam, LinearWake, Damper, WakeSourceFromFile, ResistiveWallWakeSource
from PyHEADTAIL_feedback.signal_tools.trackers_and_kickers import Tracer, AvgValueTracer, DCSuppressor, Noise
from PyHEADTAIL_feedback.processors.convolution import PhaseLinearizedLowpass, Gaussian, Sinc, Lowpass
from PyHEADTAIL_feedback.processors.register import TurnFIRFilter
from PyHEADTAIL_feedback.processors.convolution import Lowpass,  FIRFilter
from PyHEADTAIL_feedback.processors.resampling import DAC, HarmonicADC, BackToOriginalBins, Upsampler
from PyHEADTAIL_feedback.processors.resampling import Quantizer
from PyHEADTAIL_feedback.processors.addition import NoiseGenerator
#from PyHEADTAIL_feedback.core import process
from PyHEADTAIL_feedback.core import bin_mids, process
from MD4063_filter_functions import calculate_coefficients_3_tap, calculate_hilbert_notch_coefficients

class DamperImpulseResponse(object):
    """ A tracer which damps beam oscillations as a trasverse damper.
    """
    def __init__(self, turns, gain, processors, pickup_variable = 'x', kick_variable = 'x'):
        """
        Parameters
        ----------
        gain : float
            Pass band gain of the damper, i.e. 2/damping_time
        processors : list
            A list of signal processors
        pickup_variable : str
            A beam property, which is readed as a pickup signal
        kick_variable : str
            A beam property, which is kicked
        """
        self.gain = gain
        self.processors = processors
        self.pickup_variable = pickup_variable
        self.kick_variable = kick_variable
        self.turns = turns
        
        self.rotation_done = False



    def get_impulse_response(self, beam, **kwargs):

        # generates signal from the beam
        parameters, signal = beam.signal(self.pickup_variable)

        empty_signal = np.zeros(len(signal))

        turn_by_turn_impulses = []

        for i in range(self.turns+10):
            
            if i == 10:
                # processes the signal
                kick_parameters_x, kick_signal_x = process(parameters, signal, self.processors,
                                                           slice_sets=beam.slice_sets, **kwargs)
            else:
                # processes the signal
                kick_parameters_x, kick_signal_x = process(parameters, empty_signal, self.processors,
                                                           slice_sets=beam.slice_sets, **kwargs)
            if i >= 10:    
                if kick_signal_x is not None:
                    print('Non empty kick: ' + str(np.max(kick_signal_x)))
    #                print('Non empty kick!!')
                    turn_by_turn_impulses.append(np.copy(kick_signal_x))
                else:
                    print('Empty kick')
                    turn_by_turn_impulses.append(empty_signal)
        
        return turn_by_turn_impulses



class SingleBunchTracer(object):

    def __init__(self, n_turns, bunch_id):
        
        self.n_turns = n_turns
        self.bunch_id = bunch_id
        self.data = np.zeros(n_turns)
        self.counter = 0
        self.rotation_done = False
        
    @property
    def done(self):
        return False
    
    def operate(self, beam, **kwargs):
        self.data[self.counter] = beam.x[self.bunch_id]
        self.counter += 1


def run(argv):
    job_id = int(argv[0])
    case_id = 1

    intensity = 6.0e10
#    intensity = 6.0e11
    tune_error = 0.
#    damping_time = 20.
    damping_time = 50

#    for i, damping_time in enumerate(damping_times):
    gain = 2./damping_time
#    it = job_id*len(damping_times) + i
    it = job_id


    print(str(datetime.datetime.now()) + ', cycle ' + str(it) + ' -> gain = ' + str(gain) + ', tune_error = ' + str(tune_error))

    # SIMULATION PARAMETERS
    # ---------------------
    n_turns = 1

    # MACHINE PARAMETERS
    #-------------------
    

#    h_bunch = 3564

#    accQ_x = 20.13
    accQ_x = 20.18
    Q=accQ_x
    
    
    circumference = 26658.883
    
    beta_x = circumference / (2.*np.pi*accQ_x)
    n_bunches=3564

    # FILLING SCHEME
    # --------------
    filling_scheme = np.arange(n_bunches)

    # GENERATES BEAM
    # --------------
    beam = CircularPointBeam(filling_scheme, circumference, n_bunches, intensity,
                             circular_overlapping=0, n_segments = 1, beta_x = beta_x)

    impulse_idx = 1700
    beam.x[impulse_idx] =+ 1e-3

    # DAMPER SETTINGS
    # ---------------
    
    taps = np.linspace(1,64,64)
    FIR_gain_filter_gaussian = np.exp(-0.5*(taps-32.5)**2/8**2)
    FIR_gain_filter_gaussian = FIR_gain_filter_gaussian/np.sum(FIR_gain_filter_gaussian)
    
    ADC_range = (-3e-3, 3e-3)
    
    FIR_GAIN_120MSPS_20_MHz = [ -11,-3,60,114,-11,-312,-376,220,1043,790,-1118,-2875,-1196,5143,12876,16383,12876,5143,-1196,-2875,-1118,790,1043,220,-376,-312,-11,114,60,-3,-11,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    FIR_GAIN_120MSPS_20_MHz = np.array(FIR_GAIN_120MSPS_20_MHz)
    FIR_PHASE_40MSPS_6p5_MHz = [0,0,0,0,0,0,0,0,4,-5,7,-9,12,-12,23,-6,77,113,538,1509,6324,32767,-6718,-360,-180,26,-40,24,-21,16,-13,11,-9,7,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]    
    FIR_PHASE_40MSPS_6p5_MHz = np.array(FIR_PHASE_40MSPS_6p5_MHz)
    
    lowpass20MHz = [38,118,182,112,-133,-389,-385,-45,318,257,-259,-665,-361,473,877,180,-996,-1187,162,1670,1329,-954, -2648, -1219,2427,4007,419,-5623, -6590,2893,19575,32700,32700,19575, 2893,-6590,-5623,419,4007,2427,-1219,-2648, -954, 1329,1670, 162,-1187,-996,180,877,473,-361,-665,-259, 257,318,-45,-385,-389,-133,
    112,182,118,38]
    
    phaseEqualizer = [2,4,7,10,12,16,19,22,27,31,36,42,49,57,67,77,90,104,121,141,164,191,223,261,305, 358,422,498,589,700,836,1004,1215,1483,1832,2301, 2956,3944,5600,9184,25000,-16746,-4256,-2056,-1195,-769,-523,-372,-271,-202,-153, -118,-91,-71,-56,-44,-34,-27,-20,-15,-11,-7,-4,-1] 
    print('')
    print('FIR_GAIN_120MSPS_20_MHz')
    for i, val in enumerate(FIR_GAIN_120MSPS_20_MHz):
        print(str(i) + ': ' + str(val))
    print('')
    print('FIR_PHASE_40MSPS_6p5_MHz')
    for i, val in enumerate(FIR_PHASE_40MSPS_6p5_MHz):
        print(str(i) + ': ' + str(val))
    
    FIR_phase_filter = phaseEqualizer/np.sum(phaseEqualizer)

    FIR_gain_filter = lowpass20MHz/np.sum(lowpass20MHz)

    fc=1e6 # The cut off frequency of the power amplifier
    ADC_bits = 16
    
    DAC_bits = 14
    DAC_range = ADC_range
    
    delay = 1
    additional_phase = 0.25

#    turn_phase_filter_x = calculate_coefficients_3_tap(Q, delay, additional_phase)
    turn_phase_filter_x = calculate_hilbert_notch_coefficients(Q, delay, additional_phase)


    processors = [
#            HarmonicADC(f_RF, ADC_bits, ADC_range,
#                        n_extras=extra_adc_bins),
#            NoiseGenerator(input_noise),
            Quantizer(ADC_bits, ADC_range, debug=True),
            TurnFIRFilter(turn_phase_filter_x, Q, delay=1, debug=True),
#            FIRFilter(FIR_phase_filter, zero_tap = 21),
            FIRFilter(FIR_phase_filter, zero_tap = 40, debug=True),
#            Upsampler(3, [0,3,0], debug=True),
#            FIRFilter(FIR_gain_filter, zero_tap = 16),
            Upsampler(3, [0,3,0], debug=True),
#            FIRFilter(FIR_gain_filter_gaussian, zero_tap = 31),
            FIRFilter(FIR_gain_filter, zero_tap = 33, debug=True),
#            DAC(DAC_bits, DAC_range, method = ('upsampling', 1)),
            Quantizer(DAC_bits, DAC_range, debug=True),
            Lowpass(fc, f_cutoff_2nd=20*fc, debug=True),
#            BackToOriginalBins(),
#            NoiseGenerator(output_noise),
    ]
    
    
    response_calculator = DamperImpulseResponse(10, gain,processors)
    impulse_response = response_calculator.get_impulse_response(beam)
    
    
    parsed_turns = []
    for i, d in enumerate(impulse_response):
        if i == 0:
            parsed_turns.append(d[impulse_idx*3:])
        else:
            parsed_turns.append(d)
    
    parsed_impulse_response = np.concatenate(parsed_turns)
    parsed_bins = np.linspace(0,len(parsed_impulse_response)/3.-1,len(parsed_impulse_response))
    parsed_time = parsed_bins*circumference/(float(n_bunches*3)*c)
    
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10,4))
    
    ax1.plot(parsed_bins,parsed_impulse_response)
    ax2.plot(parsed_time,parsed_impulse_response)
    plt.show()

if __name__=="__main__":
	run(sys.argv[1:])
