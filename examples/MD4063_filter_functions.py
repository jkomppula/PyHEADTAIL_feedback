import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c
from scipy import signal
from PyHEADTAIL_feedback.core import Parameters, process, z_bins_to_bin_edges, bin_mids, default_macros
from PyHEADTAIL_feedback.processors.register import TurnDelay, TurnFIRFilter

def plot_filter_responses(filters, group_delays, labels, circumference, Q_x, delay, additional_phase):

    f_rev = 1./(circumference/c)

    w_h = []
    angles = []
    for i, f in enumerate(filters):
        w, h = signal.freqz(f)
        group_delay = group_delays[i]
        a = np.unwrap(np.angle(h)+group_delay*w)
        w_h.append((w,h))
        angles.append(a)
    
    lines_styles = [
        '-',
        '--',
        '-.',
        ':',
        '-',
        '--',
        '-.',
        ':',
        '-',
        '--',
        '-.',
        ':',
        '-',
    ]


    freq_conv = (Q_x%1.)*f_rev /(2.*np.pi*(Q_x%1.))


    fig = plt.figure()
    plt.title('Filter frequency responses')
    ax1 = fig.add_subplot(111)
    for i, f in enumerate(filters):
        plt.plot(freq_conv*w_h[i][0], abs(w_h[i][1]), lines_styles[i],color='b', label=labels[i])
    plt.ylabel('Amplitude', color='b')
    plt.xlabel('Frequency [Hz]')
    ax2 = ax1.twinx()
    for i, f in enumerate(filters):
        plt.plot(freq_conv*w_h[i][0], angles[i]%(2.*np.pi)*360/2./np.pi, lines_styles[i],color='g', label=labels[i])
    plt.ylabel('Angle (degrees)', color='g')
    plt.grid()
    plt.axis('tight')

    x_min = 0.8*f_rev*(Q_x%1.)
    x_max = 1.2*f_rev*(Q_x%1.)

#    x_min = 0.1*f_rev*(Q_x%1.)
#    x_max = 2.0*f_rev*(Q_x%1.)


    ax1.set_xlim(x_min, x_max)
    ax2.set_xlim(x_min, x_max)

    ax1.axvline(freq_conv*2.*np.pi*(Q_x%1.), color='k', linestyle=':')
    ax2.axhline(2.*np.pi*(delay*Q_x%1.+additional_phase)*360/2./np.pi, color='g', linestyle=':')
    ax1.axhline(1., color='b', linestyle=':')
    ax1.legend()
    plt.show()


def calculate_sparse(Q, delay, additional_phase):
    ppt = 2.* np.pi
    c12 = np.cos(Q *3.*ppt)
    s12 = np.sin(Q *3.*ppt)
    c13 = np.cos(Q *6.*ppt)
    s13 = np.sin(Q *6.*ppt)
    c14 = np.cos((Q *(6+delay)-additional_phase)*ppt)
    s14 = np.sin((Q *(6+delay)-additional_phase)*ppt)
    
    divider = -1.*(-c12*s13+c13*s12-s12+s13)

    cx1 = c14*(1-(c12*s13-c13*s12)/divider)+s14*(-c12+c13)/divider
    cx2 = (c14*(-(-s13))+s14*(-c13+1))/divider
    cx3 = (c14*(-(s12))+s14*(c12-1))/divider
    
    return cx1, cx2, cx3



def calculate_coefficients_sparse(Q, delay, additional_phase):
    c1_x, c2_x, c3_x = calculate_sparse(-Q,delay, additional_phase)

    FIR_filter_new = [
        c3_x,
        0,
        0,
        c2_x,
        0,
        0,
        c1_x,
    ]
    return FIR_filter_new

def calculate_asymmetric(Q, delay, additional_phase):
    ppt = 2.* np.pi
    c12 = np.cos(Q *3.*ppt)
    s12 = np.sin(Q *3.*ppt)
    c13 = np.cos(Q *4.*ppt)
    s13 = np.sin(Q *4.*ppt)
    c14 = np.cos((Q *(4+delay)-additional_phase)*ppt)
    s14 = np.sin((Q *(4+delay)-additional_phase)*ppt)
    
    divider = -1.*(-c12*s13+c13*s12-s12+s13)

    cx1 = c14*(1-(c12*s13-c13*s12)/divider)+s14*(-c12+c13)/divider
    cx2 = (c14*(-(-s13))+s14*(-c13+1))/divider
    cx3 = (c14*(-(s12))+s14*(c12-1))/divider
    
    return cx1, cx2, cx3



def calculate_coefficients_asymmetric(Q, delay, additional_phase):
    c1_x, c2_x, c3_x = calculate_asymmetric(-Q,delay, additional_phase)

    FIR_filter_new = [
        c3_x,
        c2_x,
        0,
        0,
        c1_x,
    ]
    return FIR_filter_new

def calculate_coefficients(Q, delay, additional_phase):
    ppt = 2.* np.pi
    c12 = np.cos(Q *1.*ppt)
    s12 = np.sin(Q *1.*ppt)
    c13 = np.cos(Q *2.*ppt)
    s13 = np.sin(Q *2.*ppt)
    c14 = np.cos((Q *(2+delay)-additional_phase)*ppt)
    s14 = np.sin((Q *(2+delay)-additional_phase)*ppt)
    
    divider = -1.*(-c12*s13+c13*s12-s12+s13)

    cx1 = c14*(1-(c12*s13-c13*s12)/divider)+s14*(-c12+c13)/divider
    cx2 = (c14*(-(-s13))+s14*(-c13+1))/divider
    cx3 = (c14*(-(s12))+s14*(c12-1))/divider
    
    return cx1, cx2, cx3

def calculate_coefficients_3_tap(Q, delay, additional_phase):
    c1_x, c2_x, c3_x = calculate_coefficients(-Q,delay, additional_phase)

    FIR_filter_new = [
        c3_x,
        c2_x,
        c1_x,
    ]
    return FIR_filter_new

def calculate_coefficients_4_tap(Q, delay, additional_phase):
    c1_x, c2_x, c3_x = calculate_coefficients(-Q,delay+1, additional_phase)

    FIR_filter_new = [
        0.,
        c3_x,
        c2_x,
        c1_x,
    ]
    return FIR_filter_new

def calculate_coefficients_long(length, Q, delay, additional_phase, use_every_n=1):
    
    total_filter = None
    
    for i in range(length):
        coeffs = calculate_coefficients_3_tap(Q, delay+3*i, additional_phase)
        coeffs = np.array(coeffs)
        if total_filter is None:
            total_filter = np.zeros(len(coeffs)*length)
        i_from = len(coeffs)*i
        i_to = len(coeffs)*i+len(coeffs)
        total_filter[i_from:i_to] = total_filter[i_from:i_to]+coeffs/float(length)
    
    print('FIR filter coefficients are: ' + str(total_filter))
    
    return total_filter

def calculate_coefficients_long2(length, Q, delay, additional_phase, use_every_n=1):
    
    total_filter = None
    
    for i in range(length):
        coeffs = calculate_coefficients_3_tap(Q, delay+i, additional_phase)
        coeffs = np.array(coeffs)
        if total_filter is None:
            total_filter = np.zeros(len(coeffs)+length-1)
        i_from = i
        i_to =i+len(coeffs)
        total_filter[i_from:i_to] = total_filter[i_from:i_to]+coeffs/float(length)
    
    print('FIR filter coefficients are: ' + str(total_filter))
    
    return total_filter

def calculate_coefficients_long_decay(length, Q, delay, additional_phase, use_every_n=1):
    
    total_filter = None
    
    x = np.linspace(0,length-1,length)
    weights = np.exp(-3.*x/float(length))
    weights = weights/np.sum(weights)
#    weights = weights[::-1]
    
    for i in range(length):
        coeffs = calculate_coefficients_3_tap(Q, delay+i, additional_phase)
        coeffs = np.array(coeffs)
        if total_filter is None:
            total_filter = np.zeros(len(coeffs)+length-1)
        i_from = i
        i_to =i+len(coeffs)
        total_filter[i_from:i_to] = total_filter[i_from:i_to]+coeffs*weights[i]
    
    print('FIR filter coefficients are: ' + str(total_filter))
    
    return total_filter


def hilbert_notch_coefficients(Q, phase_correction, gain_correction, delay, additional_phase):

    turn_notch_filter = [1,-1]
#    phase_shift_x = -((4.0+delay) * Q+additional_phase+phase_correction) * 2.* np.pi
    # 3.5 = group delay of the notch + hilbert
    # 0.25 = phase shift of the notch
    phase_shift_x = -((3.5+delay) * Q+additional_phase+phase_correction-0.25) * 2.* np.pi
    turn_phase_filter_x = [gain_correction*-2. * np.sin(phase_shift_x)/(np.pi * 3.),
                       0,
                       gain_correction*-2. * np.sin(phase_shift_x)/(np.pi * 1.),
                       gain_correction*np.cos(phase_shift_x),
                       gain_correction*2. * np.sin(phase_shift_x)/(np.pi * 1.),
                       0,
                       gain_correction*2. * np.sin(phase_shift_x)/(np.pi * 3.)
                       ]

    return np.convolve(turn_notch_filter, turn_phase_filter_x)

def calculate_hilbert_notch_coefficients(Q, delay, additional_phase):
    phase_correction, gain_correction = calculate_hilbert_corrections(Q, delay, additional_phase)
    print('phase_correction: ' + str(phase_correction))
    print('gain_correction: ' + str(gain_correction))
    return hilbert_notch_coefficients(Q, phase_correction, gain_correction, delay, additional_phase)

def calculate_hilbert_notch_coefficients_deg(angle):

    turn_notch_filter = [1,-1]
    phase_shift_x = -(angle/360.) * 2.* np.pi
    turn_phase_filter_x = [-2. * np.sin(phase_shift_x)/(np.pi * 3.),
                       0,
                       -2. * np.sin(phase_shift_x)/(np.pi * 1.),
                       np.cos(phase_shift_x),
                       2. * np.sin(phase_shift_x)/(np.pi * 1.),
                       0,
                       2. * np.sin(phase_shift_x)/(np.pi * 3.)
                       ]

    return np.convolve(turn_notch_filter, turn_phase_filter_x)

def hilbert_in_angle(angle):

    turn_notch_filter = [1,-1]
    phase_shift_x = -(angle/360.) * 2.* np.pi
    turn_phase_filter_x = [-2. * np.sin(phase_shift_x)/(np.pi * 3.),
                       0,
                       -2. * np.sin(phase_shift_x)/(np.pi * 1.),
                       np.cos(phase_shift_x),
                       2. * np.sin(phase_shift_x)/(np.pi * 1.),
                       0,
                       2. * np.sin(phase_shift_x)/(np.pi * 3.)
                       ]

    return turn_phase_filter_x

def calculate_hilbert_coefficients(Q, delay, additional_phase, phase_correction = 0., gain_correction = 1.):
    phase_shift_x = -((3+delay) * Q+additional_phase+phase_correction) * 2.* np.pi
    turn_phase_filter_x = [gain_correction*-2. * np.sin(phase_shift_x)/(np.pi * 3.),
                       0,
                       gain_correction*-2. * np.sin(phase_shift_x)/(np.pi * 1.),
                       gain_correction*np.cos(phase_shift_x),
                       gain_correction*2. * np.sin(phase_shift_x)/(np.pi * 1.),
                       0,
                       gain_correction*2. * np.sin(phase_shift_x)/(np.pi * 3.)
                       ]

    return turn_phase_filter_x

offset = 20.

n_points = 2
min_val = 0.
max_val = 10.
n_turns = 20000

def generate_signal(min_val,max_val,n_points):
    bin_width = (max_val-min_val)/float(n_points-1)
    z_bins = np.linspace(min_val-bin_width/2.,max_val+bin_width/2.,n_points+1)
    
    signal_class = 0
    bin_edges = z_bins_to_bin_edges(z_bins)
    n_segments = 1
    n_bins_per_segment = n_points
    segment_ref_points = [(min_val + max_val) / 2.]
    previous_parameters = []
    location = 0. 
    beta = 1.
    
    parameters = Parameters(signal_class, bin_edges, n_segments,
               n_bins_per_segment, segment_ref_points,
               previous_parameters, location, beta)
    signal_x = bin_mids(bin_edges)
    signal_xp = np.zeros(n_points)
    
    return parameters, signal_x, signal_xp


def rotate(angle, parameters, x, xp):
    s = np.sin(angle)
    c = np.cos(angle)
    beta_x = parameters['beta']

    new_x = c * x + beta_x * s * xp
    new_xp = (-1. / beta_x) * s * x + c * xp
    
    return parameters, new_x, new_xp

def process_signals(processors, parameters, x, xp):
    parameters_out, x_out = process(parameters, x, processors)
    return x_out
    
    
def track_signal(n_turns, gain, Q_x, processors, parameters, signal_x, signal_xp):
    signals = np.zeros(n_turns)
    corrections = np.zeros(n_turns)

    for i in range(n_turns):
        angle = Q_x*2.*np.pi
        parameters, signal_x, signal_xp = rotate(angle,parameters, signal_x, signal_xp)
        correction = process_signals(processors, parameters, signal_x, signal_xp)
        signal_xp = signal_xp - gain*correction
        signals[i] = signal_x[-1]
        corrections[i] = correction[-1]
    return signals, corrections

class Offset(object):
    def __init__(self, offset, **kwargs):
        self._offset = offset
        
        self.extensions = []
        self._macros = [] + default_macros(self, 'Offset', **kwargs)
        
        
    def process(self, parameters, signal, *args, **kwargs):
            
        return parameters, signal + self._offset

from scipy.optimize import minimize
def def_error_function(Q_x, delay, additional_phase):
    FIR_ref = calculate_coefficients_3_tap(Q_x, delay, additional_phase)
    
    processors_ref = [
        Offset(offset),
        TurnFIRFilter(FIR_ref, Q_x, delay=delay),
    ]
    
    parameters, signal_x, signal_xp = generate_signal(min_val, max_val, n_points)
    signal_ref, correction_ref = track_signal(100, 0., Q_x, processors_ref, parameters, signal_x, signal_xp)

    def error_function(coord):
            min_val = 0.
            max_val = 10.
            n_points = 2
            dGain = coord[1]
            dQ = coord[0]
            FIR_filter = hilbert_notch_coefficients(Q_x, dQ, dGain, delay,additional_phase)

            processors = [
                Offset(offset),
                TurnFIRFilter(FIR_filter, Q_x, delay=delay),
            ]

            parameters, signal_x, signal_xp = generate_signal(min_val, max_val, n_points)

            signal_classic, correction_classic = track_signal(100, 0., Q_x, processors, parameters, signal_x, signal_xp)
        #     print np.sum(np.abs(correction_classic[-20:]-correction_ref[-20:]))
        #     print 'dQ:' + str(dQ) + ', dGain:' + str(dGain) + ' -> ' + str(np.sum(np.abs(correction_classic[-20:]-correction_ref[-20:])))
            return np.sum(np.abs(correction_classic[-20:]-correction_ref[-20:]))
    return error_function

def calculate_hilbert_corrections(Q, delay, additional_phase):
        
    error_function = def_error_function(Q, delay, additional_phase)

    initial_guess = np.array([0.,1.0])
    bnds = ((-0.25, 0.5), (0.25, 1.5))
    res = minimize(error_function,initial_guess,method='TNC', bounds=bnds, tol=1e-7)
    print(res)
    return res.x[0], res.x[1]