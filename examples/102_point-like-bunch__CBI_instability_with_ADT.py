from __future__ import division
import sys, os
import sys
# PyHEADTAIL location if it's not already in the PYTHONPATH environment variable
sys.path.append('../../')
import matplotlib.pyplot as plt

import sys
import datetime
import numpy as np
from scipy.constants import c, e
from PyHEADTAIL_feedback.signal_tools.signal_generators import SimpleBeam, CircularPointBeam
from PyHEADTAIL_feedback.signal_tools.trackers_and_kickers import LinearWake, Damper, WakeSourceFromFile, ResistiveWallWakeSource
from PyHEADTAIL_feedback.signal_tools.trackers_and_kickers import Tracer, AvgValueTracer, DCSuppressor, Noise, track_beam
from PyHEADTAIL_feedback.processors.convolution import PhaseLinearizedLowpass, Gaussian, Sinc, Lowpass
from PyHEADTAIL_feedback.processors.register import TurnFIRFilter
from PyHEADTAIL_feedback.processors.convolution import Lowpass,  FIRFilter
from PyHEADTAIL_feedback.processors.resampling import DAC, HarmonicADC, BackToOriginalBins, Upsampler
from PyHEADTAIL_feedback.processors.resampling import Quantizer
from PyHEADTAIL_feedback.processors.addition import NoiseGenerator
from MD4063_filter_functions import calculate_coefficients_3_tap, calculate_hilbert_notch_coefficients

def run(argv):
    np.random.seed(0)
    job_id = int(argv[0])
    case_id = 0

    intensity = 1.1e11
    damping_time = 50.
    
    it = job_id

    # SIMULATION PARAMETERS
    # ---------------------
    n_turns = 5000
    n_traces = 10

    # MACHINE PARAMETERS
    #-------------------
    circumference = 26658.883
    p0 = 450e9 * e / c
    accQ_x = 59.31
    
    beta_x = circumference / (2.*np.pi*accQ_x)
    n_bunches=3564
    
    filling_pattern_array = [
            # 3x[(72b + 8e) x 3 + 30 e]
            (72, 1),
            (8, 0),
            (72, 1),
            (8, 0),
            (72, 1),
            (8, 0),
            (30, 0),
            # 
            (72, 1),
            (8, 0),
            (72, 1),
            (8, 0),
            (72, 1),
            (8, 0),
            (30, 0),
            # 
            (72, 1),
            (8, 0),
            (72, 1),
            (8, 0),
            (72, 1),
            (8, 0),
            (30, 0),
            # 
            # 1e
            (1, 0),
            # 
            # [(72b + 8e) x 3 + 30 e + (72b + 8e) x 3 + 30 e + (72b + 8e) x 4 + 31 e]
            (72, 1),
            (8, 0),
            (72, 1),
            (8, 0),
            (72, 1),
            (8, 0),
            (30, 0),
            # 
            (72, 1),
            (8, 0),
            (72, 1),
            (8, 0),
            (72, 1),
            (8, 0),
            (30, 0),
            # 
            (72, 1),
            (8, 0),
            (72, 1),
            (8, 0),
            (72, 1),
            (8, 0),
            (72, 1),
            (8, 0),
            (31, 0),
            # 
            # 
            (72, 1),
            (8, 0),
            (72, 1),
            (8, 0),
            (72, 1),
            (8, 0),
            (30, 0),
            # 
            (72, 1),
            (8, 0),
            (72, 1),
            (8, 0),
            (72, 1),
            (8, 0),
            (30, 0),
            # 
            (72, 1),
            (8, 0),
            (72, 1),
            (8, 0),
            (72, 1),
            (8, 0),
            (72, 1),
            (8, 0),
            (31, 0),
            #
            #
            (72, 1),
            (8, 0),
            (72, 1),
            (8, 0),
            (72, 1),
            (8, 0),
            (30, 0),
            # 
            (72, 1),
            (8, 0),
            (72, 1),
            (8, 0),
            (72, 1),
            (8, 0),
            (30, 0),
            # 
            (72, 1),
            (8, 0),
            (72, 1),
            (8, 0),
            (72, 1),
            (8, 0),
            (72, 1),
            (8, 0),
            (31, 0),
            #
            #
            (80, 0),
            
            ]
    
    
#    filling_scheme = np.arange(n_bunches)
    filling_scheme = []
    counter = 0
    for s in filling_pattern_array:
        for i in range(s[0]):
            if s[1] == 1:
                filling_scheme.append(counter)
            counter += 1


    
    # WAKE SETTINGS
    # =============
    wakefile = 'wakeforhdtl_PyZbase_Allthemachine_450GeV_B1_LHC_inj_450GeV_B1.dat'
    n_turns_wake = 30

    
    wake_source= WakeSourceFromFile(wakefile,0,2)
    wake = LinearWake(wake_source, n_turns_wake)
 
    # GENERATES BEAM
    # --------------
    beam = CircularPointBeam(filling_scheme, circumference, n_bunches, intensity,
                             circular_overlapping=2, n_segments = 9, beta_x = beta_x)

    beam.set_beam_paramters(p0)
    beam.init_noise(1e-7)
    amplitude = 2e-6
    
    # Data
    # ---------------
    triggers = [
        ('turn', n_turns-n_traces),
        ('mean_abs_x', 0.3e-2)
    ]


    avg_data = AvgValueTracer(n_turns)
    bunch_by_bunch_data = Tracer(n_traces, variables=['x', 'xp'], triggers=triggers)


    # DAMPER SETTINGS
    # ---------------
    
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
    ADC_range = (-1., 1.)
    
    # signal processing delay in turns before the first measurements is applied
    delay = 1
    
    additional_phase = 0.

    turn_phase_filter_x = calculate_coefficients_3_tap(accQ_x, delay, additional_phase)
#    turn_phase_filter_x = calculate_hilbert_notch_coefficients(Q, delay, additional_phase)

    
    # The LHC ADT model
    # NOTE: correct timing of the system depends on the used filters
    #       * correct timing means that the peak of the impulse response is exactly 1, 2, etc turns after the impulse
    #       * coarse timing can be adjusted with zero_tap parameter of the FIR_gain_filter
    #       * fine timing can be adjusted with weigth list of the Upsampler (the sum of the weights must be)
    
    processors = [
            Quantizer(ADC_bits, ADC_range),
            TurnFIRFilter(turn_phase_filter_x, accQ_x, delay=1),
            FIRFilter(FIR_phase_filter, zero_tap=40),
            Upsampler(3, [1.5,1.5,0]),
            FIRFilter(FIR_gain_filter, zero_tap=34),
            Quantizer(ADC_bits, ADC_range),
            Lowpass(fc, f_cutoff_2nd=20*fc),
            BackToOriginalBins(),
    ]
    
    
    gain = 2./damping_time
    damper = Damper(gain,processors)

    

    trackers = [
            damper,
            wake,
            avg_data,
            bunch_by_bunch_data
    ]
        
        
        
    # GENERATES OUTPUT DATA FILES
    # ---------------------------

    track_beam(beam, trackers, n_turns, accQ_x)

#    avg_data.save_to_file('case_' + str(it))
#    bunch_by_bunch_data.save_to_file('case_' + str(it))
    
#    %matplotlib inline
    import matplotlib.pyplot as plt

    n_points = 100
    m = avg_data.epsn_x
    m = (m>0)
    d = avg_data.mean_abs_x
    d = d[m]
    # print d 
    turns = np.linspace(1,len(d),len(d))
    coeffs_x = np.polyfit(turns[-n_points:], np.log(d[-n_points:]),1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax2.semilogy(turns, d)
    ax2.semilogy(turns, np.exp(coeffs_x[0]*turns + coeffs_x[1]),label='Growth time ' + str(int(1/coeffs_x[0])) + ' turns')
    ax2.set_ylim(0.5*np.min(d),2*np.max(d))    
    ax2.set_xlabel('Turn')
    ax2.set_ylabel('|Beam avg oscillation|')
    ax2.legend()

    
    colors = [plt.cm.viridis(i) for i in np.linspace(0, 1, n_traces)]
    for i in range(3):
        ax1.plot(bunch_by_bunch_data.x[i], '-', color=colors[i], label='Turn ' + str(int(len(turns)-n_traces-1+i)))

    ax1.set_xlabel('Bucket #')
    ax1.set_ylabel('BPM signal')
    ax1.legend()

    plt.show()

if __name__=="__main__":
	run(sys.argv[1:])
