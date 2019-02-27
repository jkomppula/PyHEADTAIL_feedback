from __future__ import division
import sys, os
BIN = os.path.expanduser("../../")
sys.path.append(BIN)
import sys
import numpy as np
import matplotlib.pyplot as plt
from PyHEADTAIL_feedback.processors.register import TurnFIRFilter
from PyHEADTAIL_feedback.processors.convolution import Lowpass,  FIRFilter
from PyHEADTAIL_feedback.processors.resampling import DAC, HarmonicADC, BackToOriginalBins, Upsampler
from PyHEADTAIL_feedback.processors.resampling import Quantizer
from MD4063_filter_functions import calculate_coefficients_3_tap, calculate_hilbert_notch_coefficients
from PyHEADTAIL_feedback.signal_tools.response_tools import DamperImpulseResponse


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


def run():

    # MACHINE PARAMETERS
    #-------------------
    
    Q=59.31
    circumference = 26658.883
    beta_beam = 1.
    n_bunches=3564

    # DAMPER SETTINGS
    # ---------------
    
    taps = np.linspace(1,64,64)
    FIR_gain_filter_gaussian = np.exp(-0.5*(taps-32.5)**2/8**2)
    FIR_gain_filter_gaussian = FIR_gain_filter_gaussian/np.sum(FIR_gain_filter_gaussian)

    
    lowpass100kHz = [1703, 1169, 1550, 1998, 2517, 3108, 3773, 4513, 5328, 6217, 7174, 8198, 9282, 10417, 11598, 12813, 14052, 15304, 16555, 17793, 19005, 20176, 21294, 22345, 23315, 24193, 24969, 25631, 26171, 26583, 26860, 27000, 27000, 26860, 26583, 26171, 25631, 24969, 24193, 23315, 22345, 21294, 20176, 19005, 17793, 16555, 15304, 14052, 12813, 11598, 10417, 9282, 8198, 7174, 6217, 5328, 4513, 3773, 3108, 2517, 1998, 1550, 1169, 1703]
    
    lowpassEnhanced = [490,177,-478,-820,-370,573,1065,428,-909, -1632,-799,1015, 2015,901,-1592,-3053,-1675,1642, 3670,1841,-2828,-6010,-3929,2459,7233,4322,-6384,-17305,-18296,-5077,16097,32000, 32000,16097,-5077,-18296,-17305,-6384,4322, 7233,2459,-3929,-6010,-2828,1841,3670,1642,-1675,-3053,-1592,901,2015,1015, -799,-1632,-909,428,1065,573,-370,-820,-478,177,490]
    
    lowpass20MHz = [38,118,182,112,-133,-389,-385,-45,318,257,-259,-665,-361,473,877,180,-996,-1187,162,1670,1329,-954, -2648, -1219,2427,4007,419,-5623, -6590,2893,19575,32700,32700,19575, 2893,-6590,-5623,419,4007,2427,-1219,-2648, -954, 1329,1670, 162,-1187,-996,180,877,473,-361,-665,-259, 257,318,-45,-385,-389,-133,
    112,182,118,38]
    
    phaseEqualizer = [2,4,7,10,12,16,19,22,27,31,36,42,49,57,67,77,90,104,121,141,164,191,223,261,305, 358,422,498,589,700,836,1004,1215,1483,1832,2301, 2956,3944,5600,9184,25000,-16746,-4256,-2056,-1195,-769,-523,-372,-271,-202,-153, -118,-91,-71,-56,-44,-34,-27,-20,-15,-11,-7,-4,-1] 
        
    FIR_phase_filter = np.loadtxt('./injection_error_input_data/FIR_Phase_40MSPS.csv')
    FIR_phase_filter = np.array(phaseEqualizer)    
    FIR_phase_filter = FIR_phase_filter/float(np.sum(FIR_phase_filter))

    FIR_gain_filter = np.array(lowpass20MHz)
    FIR_gain_filter = FIR_gain_filter/float(np.sum(lowpass20MHz))


     # Cut-off frequency of the kicker system
    fc=1e6
    ADC_bits = 16 
    ADC_range = (-1e-3, 1e-3)
    
    # signal processing delay in turns before the first measurements is applied
    delay = 1
    
    # betatron phase advance between the pickup and the kicker. The value 0.25 
    # corresponds to the 90 deg phase change from from the pickup measurements
    # in x-plane to correction kicks in xp-plane.
    additional_phase = 0.25

#    turn_phase_filter_x = calculate_coefficients_3_tap(Q, delay, additional_phase)
    turn_phase_filter_x = calculate_hilbert_notch_coefficients(Q, delay, additional_phase)

    
    # The LHC ADT model
    # NOTE: correct timing of the system depends on the used filters
    #       * correct timing means that the peak of the impulse response is exactly 1, 2, etc turns after the impulse
    #       * coarse timing can be adjusted with zero_tap parameter of the FIR_gain_filter
    #       * fine timing can be adjusted with weigth list of the Upsampler (the sum of the weights must be)
    
    processors = [
            Quantizer(ADC_bits, ADC_range),
            TurnFIRFilter(turn_phase_filter_x, Q, delay=1),
            FIRFilter(FIR_phase_filter, zero_tap=40),
            Upsampler(3, [1.5,1.5,0]),
            FIRFilter(FIR_gain_filter, zero_tap=34),
            Quantizer(ADC_bits, ADC_range),
            # Adds more samples before the low pass filter in order to get better output sampling
            Upsampler(5, [1,1,1,1,1]), 
            Lowpass(fc, f_cutoff_2nd=20*fc),
    ]
    
    impulse_length = 10
    response_calculator = DamperImpulseResponse(processors, circumference, n_bunches, beta_beam, impulse_length, wait_before=10)
    parsed_impulse_response, samples, turns, time = response_calculator.get_impulse_response()
    
        
    fig, ax1 = plt.subplots(1,1, figsize=(6,4))
    ax1.plot(time,parsed_impulse_response)
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Normalized impulse response')
    
    ax2 = ax1.twiny()
    ax3 = ax1.twiny()
    ax1.set_xlim(np.min(time), np.max(time))
    ax2.set_xlim(np.min(turns), np.max(turns))
    ax3.set_xlim(np.min(samples), np.max(samples))
    ax2.set_xlabel('Turn')
    ax3.set_xlabel('Bucket')


    ax3.xaxis.set_ticks_position('top')
    ax3.xaxis.set_label_position('top')
    
    ax2.xaxis.set_ticks_position('top')
    ax2.xaxis.set_label_position('top')
    ax2.spines['top'].set_position(('outward', 36))

    plt.tight_layout()
    plt.show()

if __name__=="__main__":
	run()
