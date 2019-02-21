from __future__ import division

import sys, time
import numpy as np
from mpi4py import MPI
from scipy.constants import c, e, m_p

from PyHEADTAIL.particles.slicing import UniformBinSlicer
from PyHEADTAIL.impedances.wakes import CircularResonator, WakeField, WakeTable
from PyHEADTAIL.machines.synchrotron import Synchrotron
from PyHEADTAIL_feedback.feedback import OneboxFeedback
from PyHEADTAIL_feedback.processors.multiplication import ChargeWeighter
from PyHEADTAIL_feedback.processors.convolution import Gaussian
# from PyHEADTAIL_feedback.processors.linear_transform import Averager
from PyHEADTAIL_feedback.processors.misc import Bypass
from PyHEADTAIL_feedback.processors.register import Register, TurnDelay, TurnFIRFilter
from PyHEADTAIL_feedback.processors.convolution import Lowpass, FIRFilter
from PyHEADTAIL_feedback.processors.resampling import DAC, HarmonicADC, BackToOriginalBins, Upsampler
from PyHEADTAIL_feedback.processors.addition import NoiseGenerator

def hilbert_notch_coefficients(Q, phase_correction, gain_correction, delay, additional_phase):

    turn_notch_filter = [1,-1]
#    phase_shift_x = -((4.0+delay) * Q+additional_phase+phase_correction) * 2.* np.pi
    # 3.5 = group delay of the notch + hilbert
    # 0.25 = phase shift of the notch
    phase_shift_x = -((3.5+delay) * Q+additional_phase+phase_correction+0.25) * 2.* np.pi
    turn_phase_filter_x = [gain_correction*-2. * np.sin(phase_shift_x)/(np.pi * 3.),
                       0,
                       gain_correction*-2. * np.sin(phase_shift_x)/(np.pi * 1.),
                       gain_correction*np.cos(phase_shift_x),
                       gain_correction*2. * np.sin(phase_shift_x)/(np.pi * 1.),
                       0,
                       gain_correction*2. * np.sin(phase_shift_x)/(np.pi * 3.)
                       ]

    return np.convolve(turn_notch_filter, turn_phase_filter_x)

def run(argv):
    job_id = int(argv[0])
    
    test = False

    intensity = 3e11
    
    chroma = 0

    gains = np.logspace(np.log10(1e-3), np.log10(0.5),20)
    
    gain = gains[job_id]

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    print('I am rank ' + str(rank) + ' out of ' + str(size))
    print('  ')
    np.random.seed(0+rank)

       # BEAM AND MACHNINE PARAMETERS
    # ============================

    if test:
        n_macroparticles = 1000
    else:
        n_macroparticles = 70000

    charge = e
    mass = m_p
    alpha = 1.01354e-4

    p0 = 3300e9 * e / c

    accQ_x = 111.28
    accQ_y = 109.31
    Q_s = 2.76754e-3
    circumference = 97749.14
    s = None
    alpha_x = None
    alpha_y = None
    beta_x = circumference / (2.*np.pi*accQ_x)
    beta_y = circumference / (2.*np.pi*accQ_y)
    D_x = 0
    D_y = 0
    optics_mode = 'smooth'
    name = None
    n_segments = 1

    # detunings
    Qp_x = chroma
    Qp_y = chroma


    app_x = 0
    app_y = 0
    app_xy = 0

    longitudinal_mode = 'linear'

#    h_RF = 2748
    if test:
        h_RF = 1306
    else:
        h_RF = 13068
    h_bunch = h_RF
#    h_RF = 274
    wrap_z = False

    machine = Synchrotron(
            optics_mode=optics_mode, circumference=circumference,
            n_segments=n_segments, s=s, name=name,
            alpha_x=alpha_x, beta_x=beta_x, D_x=D_x,
            alpha_y=alpha_y, beta_y=beta_y, D_y=D_y,
            accQ_x=accQ_x, accQ_y=accQ_y, Qp_x=Qp_x, Qp_y=Qp_y,
            app_x=app_x, app_y=app_y, app_xy=app_xy,
            alpha_mom_compaction=alpha, longitudinal_mode=longitudinal_mode,
            h_RF=np.atleast_1d(h_RF), p0=p0,
            charge=charge, mass=mass, wrap_z=wrap_z, Q_s=Q_s)
#    machine.one_turn_map = machine.one_turn_map[1:]
#    machine.one_turn_map = [machine.one_turn_map[0]]
    print(machine.one_turn_map)

    # FILLING SCHEME
    # --------------
    filling_scheme = []
    if test:
        n_fills = 13
    else:
        n_fills = 130
#    n_fills = 13*10
    fill_gap = 17
    batch_gap = 0
    bunches_per_batch = 80
    batches_per_fill = 1

    batch_length = bunches_per_batch+batch_gap
    fill_length = fill_gap + batches_per_fill*bunches_per_batch + (batches_per_fill - 1) * batch_gap

    for i in range(n_fills):
        for j in range(batches_per_fill):
            for k in range(bunches_per_batch):
                idx = i*fill_length + j*batch_length + k
                filling_scheme.append(idx)

    # BEAM
    # ====
    epsn_x  = 2.2e-6
    epsn_y  = 2.2e-6
    sigma_z = 0.08
    allbunches = machine.generate_6D_Gaussian_bunch(
        n_macroparticles, intensity, epsn_x, epsn_y, sigma_z=sigma_z,
        filling_scheme=filling_scheme, matched=False)

    # CREATE BEAM SLICERS
    # ===================
    slicer_for_wakefields = UniformBinSlicer(20, z_cuts=(-2.*sigma_z, 2.*sigma_z), 
                                     circumference=machine.circumference, 
                                     h_bunch=h_RF)

    # WAKE PARAMETERS
    # ============================
    mpi_settings = 'linear_mpi_full_ring_fft'
    wakefile1        = './FCC_wakes_2018_09_3p3_TeV.dat'
    wake_table1          = WakeTable(wakefile1,
                                     ['time', 'dipole_x', 'dipole_y'],
                                     n_turns_wake=40)

#    wake_field           = WakeField(slicer_for_wakefields, wake_table1,
#                               circumference=machine.circumference,h_rf=int(h), h_bunch=int(h/10), mpi='memory_optimized')

    wake_field           = WakeField(slicer_for_wakefields, wake_table1, mpi=mpi_settings, Q_x=accQ_x, Q_y=accQ_y, beta_x=beta_x, beta_y=beta_y)

    machine.one_turn_map.append(wake_field)

    # SIMULATION PARAMETERS
    # ============================
    if test:
        n_turns = 25
    else:
        n_turns = 30000

    # FEEDBACK
    # ========
    
#    feedback_gain = 2./50.
#    f_c = 1e6
    
    FIR_phase_filter = np.loadtxt('./injection_error_input_data/FIR_Phase_40MSPS.csv')
    FIR_phase_filter = FIR_phase_filter/sum(FIR_phase_filter)
#    plot_FIR_coefficients(FIR_phase_filter)
    
    FIR_gain_filter = np.loadtxt('./injection_error_input_data/FIR_Gain_120MSPS.csv')
    FIR_gain_filter = FIR_gain_filter/sum(FIR_gain_filter)
#    plot_FIR_coefficients(FIR_gain_filter)

#    turn_notch_filter = [-1.,1.]
    
#    phase_shift_x = -5. * machine.Q_x * 2.* np.pi
#    turn_phase_filter_x = [-2. * np.sin(phase_shift_x)/(np.pi * 3.),
#                       0,
#                       -2. * np.sin(phase_shift_x)/(np.pi * 1.),
#                       np.cos(phase_shift_x),
#                       2. * np.sin(phase_shift_x)/(np.pi * 1.),
#                       0,
#                       2. * np.sin(phase_shift_x)/(np.pi * 3.)
#                       ]
    
#    phase_shift_y = -5. * machine.Q_y * 2.* np.pi
#    turn_phase_filter_y = [-2. * np.sin(phase_shift_y)/(np.pi * 3.),
#                       0,
#                       -2. * np.sin(phase_shift_y)/(np.pi * 1.),
#                       np.cos(phase_shift_y),
#                       2. * np.sin(phase_shift_y)/(np.pi * 1.),
#                       0,
#                       2. * np.sin(phase_shift_y)/(np.pi * 3.)
#                       ]
    
    
#    turn_FIR_filter_x = np.convolve(turn_notch_filter, turn_phase_filter_x)
#    turn_FIR_filter_y = np.convolve(turn_notch_filter, turn_phase_filter_y)
    
    phase_correction_x = 0.00763262120033
    gain_correction_x = 0.666194434134
    phase_correction_y = -0.000117213341179
    gain_correction_y = 0.602766077627
    turn_FIR_filter_x = hilbert_notch_coefficients(accQ_x, phase_correction_x, gain_correction_x, 1., 0.)
    turn_FIR_filter_y = hilbert_notch_coefficients(accQ_y, phase_correction_y, gain_correction_y, 1., 0.)

    # feedback settings
    fc=1e6 # The cut off frequency of the power amplifier
    ADC_bits = 16
    ADC_range = (-3e-3,3e-3)
    
    DAC_bits = 14
    DAC_range = (-3e-3,3e-3)
    f_bunch = 1./(circumference/c/(float(h_bunch)))
    extra_adc_bins = 10
    
    processors_detailed_x = [
        Bypass(),
        ChargeWeighter(normalization = 'segment_average'),
        HarmonicADC(f_bunch, ADC_bits, ADC_range,
                    n_extras=extra_adc_bins),
        TurnFIRFilter(turn_FIR_filter_x, machine.Q_x, delay=1),
        FIRFilter(FIR_phase_filter, zero_tap = 40),
        Upsampler(3, [0,3,0]),
        FIRFilter(FIR_gain_filter, zero_tap = 32),
        DAC(ADC_bits, ADC_range),
        Lowpass(fc, f_cutoff_2nd=20*fc),
        BackToOriginalBins(),
    ]

    processors_detailed_y = [
            Bypass(),
            ChargeWeighter(normalization = 'segment_average'),
            HarmonicADC(f_bunch, ADC_bits, ADC_range,
                        n_extras=extra_adc_bins),
            TurnFIRFilter(turn_FIR_filter_y, machine.Q_y, delay = 1),
            FIRFilter(FIR_phase_filter, zero_tap = 40),
            Upsampler(3, [0,3,0]),
            FIRFilter(FIR_gain_filter, zero_tap = 32),
            DAC(DAC_bits, DAC_range),
            Lowpass(fc, f_cutoff_2nd=20*fc),
            BackToOriginalBins(),
    ]
    
    feedback_map = OneboxFeedback(gain, slicer_for_wakefields,
                                  processors_detailed_x,processors_detailed_y,
                                  mpi=True)
    machine.one_turn_map.append(feedback_map)


    # CREATE MONITORS
    # ===============
    # bucket = machine.longitudinal_map.get_bucket(bunch)
    n_traces = 10
    
    outputpath = './Data'
    outputpath = 'Data'

    # TRACKING LOOP
    # =============
#    machine.one_turn_map.append(wake_field)
    print('\n--> Begin tracking...\n')

    import PyHEADTAIL.mpi.mpi_data as mpiTB    
    n_total_bunches = float(len(filling_scheme))
    

    if rank == 0:
        data = np.zeros((n_turns,7))
        data_x = np.zeros((n_traces,len(filling_scheme)))
        data_xp = np.zeros((n_traces,len(filling_scheme)))
        data_y = np.zeros((n_traces,len(filling_scheme)))
        data_yp = np.zeros((n_traces,len(filling_scheme)))
        
    trace_counter = 0
    trace_on = False


    for i in range(n_turns):
        t0 = time.clock()
	
        machine.track(allbunches)
        bunch_list = allbunches.split_to_views()
        
        my_abs_mean_x = 0.
        my_abs_mean_y = 0.
        my_mean_x = 0.
        my_mean_y = 0.
        my_epsn_x = 0.
        my_epsn_y = 0.
        
        for b in bunch_list:
            my_abs_mean_x += np.abs(b.mean_x())/n_total_bunches
            my_abs_mean_y += np.abs(b.mean_y())/n_total_bunches
            my_mean_x += b.mean_x()/n_total_bunches
            my_mean_y += b.mean_y()/n_total_bunches
            my_epsn_x += b.epsn_x()/n_total_bunches
            my_epsn_y += b.epsn_y()/n_total_bunches
        
        total_abs_mean_x = np.sum(mpiTB.share_numbers(my_abs_mean_x))
        total_abs_mean_y = np.sum(mpiTB.share_numbers(my_abs_mean_y))
        total_mean_x = np.sum(mpiTB.share_numbers(my_mean_x))
        total_mean_y = np.sum(mpiTB.share_numbers(my_mean_y))
        total_epsn_x = np.sum(mpiTB.share_numbers(my_epsn_x))
        total_epsn_y = np.sum(mpiTB.share_numbers(my_epsn_y))
            
        if rank == 0:
            data[i,0] = i
            data[i,1] = total_abs_mean_x
            data[i,2] = total_abs_mean_y
            data[i,3] = total_mean_x
            data[i,4] = total_mean_y
            data[i,5] = total_epsn_x
            data[i,6] = total_epsn_y
            
            if i%100 == 0:
                np.savetxt(outputpath+'/job_{:d}.dat'.format(job_id), data)
            
            if i%10 == 0:
                print('Bunch: {:4d} \t {:+3e} \t {:+3e} \t {:+3e} \t {:3e} \t {:3s}'.format(i, total_mean_x, total_mean_y, total_abs_mean_x,total_abs_mean_y, str(time.clock() - t0)))
        
        if i > (n_turns - n_traces - 2):
            trace_on = True
        
        if (total_abs_mean_x > 1e-2) and (total_abs_mean_y > 1e-2):
            trace_on = True
        
        if (total_abs_mean_x > 1e10) or (total_abs_mean_y > 1e10):
            trace_on = True
            
        if trace_on:
            my_mean_x = []
            my_mean_xp = []
            my_mean_y = []
            my_mean_yp = []
            
            for b in bunch_list:
                my_mean_x.append(b.mean_x())
                my_mean_xp.append(b.mean_x())
                my_mean_y.append(b.mean_y())
                my_mean_yp.append(b.mean_yp())
            
            
            total_mean_x = mpiTB.share_arrays(np.array(my_mean_x))
            total_mean_xp = mpiTB.share_arrays(np.array(my_mean_xp))
            total_mean_y = mpiTB.share_arrays(np.array(my_mean_y))
            total_mean_yp = mpiTB.share_arrays(np.array(my_mean_yp))
            
            if rank == 0:
                np.copyto(data_x[trace_counter,:], np.array(total_mean_x))
                np.copyto(data_xp[trace_counter,:], np.array(total_mean_xp))
                np.copyto(data_y[trace_counter,:], np.array(total_mean_y))
                np.copyto(data_yp[trace_counter,:], np.array(total_mean_yp)) 
                
            trace_counter += 1
            
        
        if trace_counter > (n_traces - 1):
            break

    print('\n*** Successfully completed!')
    if rank == 0:
        np.savetxt(outputpath+'/job_{:d}.dat'.format(job_id), data)
        np.savetxt(outputpath+'/job_{:d}_bunch_by_bunch_x.dat'.format(job_id), data_x)
        np.savetxt(outputpath+'/job_{:d}_bunch_by_bunch_xp.dat'.format(job_id), data_xp)
        np.savetxt(outputpath+'/job_{:d}_bunch_by_bunch_y.dat'.format(job_id), data_y)
        np.savetxt(outputpath+'/job_{:d}_bunch_by_bunch_yp.dat'.format(job_id), data_yp)
if __name__=="__main__":
	run(sys.argv[1:])
