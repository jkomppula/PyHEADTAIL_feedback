import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, pi
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection

def plot_3D_traces(traces, var='x'):

    data =  getattr(traces,var)

    n_turns = len(data)
    n_points = len(data[0])
    z = traces.z[0]
    turns = np.arange(n_turns)
    s_z = z.size
    s_turns = turns.size

    z = np.tile(z, (s_turns, 1))
    turns = np.tile(turns, (s_z, 1)).T

    plot_data = np.zeros((n_turns,n_points))
    for i, d in enumerate(data):
        plot_data[i,:] = d

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1, projection='3d')
    seq_col_brew = sns.color_palette("Blues_r", n_turns)
    sns.set_palette(seq_col_brew)

    surf = ax.plot_wireframe(z, turns, plot_data)
    ax.invert_yaxis()
    ax.set_xlabel('Z [m]')
    ax.set_ylabel('Turn')
    ax.set_zlabel('Variable: ' + var)
    plt.show()
    return fig, ax


def plot_traces(traces,var = 'x', mark = '.'):

    n_turns = traces.z.shape[0]
    fig = plt.figure(figsize=(8, 10))
    seq_col_brew = sns.color_palette("Blues", n_turns)
    sns.set_palette(seq_col_brew)

    ax1 = fig.add_subplot(211)
    ax11 = ax1.twiny()


    for i in xrange(n_turns):
        z=traces.z[i,:]
        t = z/c
        d = getattr(traces,var)[i,:]
        ax1.plot(t*1e9,d,mark)
        ax11.plot(z, np.zeros(len(z)))
        ax11.cla()

    ax1.set_xlabel('Time [ns]')
    ax1.set_ylabel('Signal [mm]')
    ax11.set_xlabel('Distance [m]')

    plt.show()


def plot_beam(beam,var = 'x', mark = '.', fig = None, ax1 = None, ax11 = None, label = ' '):

    if fig is None:
        fig = plt.figure(figsize=(8, 10))
    seq_col_brew = sns.color_palette("Blues_r")
    sns.set_palette(seq_col_brew)

    if ax1 is None:
        ax1 = fig.add_subplot(211)
    if ax11 is None:
        ax11 = ax1.twiny()

    z=beam.z
    t = z/c
    d = getattr(beam,var)
    if var in ['x', 'y', 'x_amp', 'y_amp', 'x_fixed', 'y_fixed']:
        d=d*1e3
        ax1.set_ylabel('Signal [mm]')
    elif var in ['xp', 'yp', 'xp_amp', 'yp_amp', 'xp_fixed', 'yp_fixed']:
        d=d*1e6
        ax1.set_ylabel('Signal [mm mrad]')
    ax1.plot(t*1e9,d,mark,label=label)
    ax11.plot(z, np.zeros(len(z)))
    ax11.cla()

    ax1.set_xlabel('Time [ns]')
    ax11.set_xlabel('Distance [m]')

#    plt.show()
    return fig, ax1, ax11


def plot_beams(beams, labels, var = 'x', mark = '.', fig = None, ax1 = None, ax11 = None):

    if fig is None:
        fig = plt.figure(figsize=(8, 10))
    seq_col_brew = sns.color_palette("Blues_r")
    sns.set_palette(seq_col_brew)

    if ax1 is None:
        ax1 = fig.add_subplot(211)
    if ax11 is None:
        ax11 = ax1.twiny()

    for beam, label in zip(beams,labels):
        z=beam.z[3:]
        t = z/c
        d = getattr(beam,var)
        d= d[3:]
        if var in ['x', 'y', 'x_amp', 'y_amp', 'x_fixed', 'y_fixed']:
            d=d*1e3
            ax1.set_ylabel('Signal [mm]')
        elif var in ['xp', 'yp', 'xp_amp', 'yp_amp', 'xp_fixed', 'yp_fixed']:
            d=d*1e6
            ax1.set_ylabel('Signal [mm mrad]')
        ax1.plot(t*1e9,d,mark,label=label)
        ax11.plot(z, np.zeros(len(z)))
        ax11.cla()

    ax1.set_xlabel('Time [ns]')
    ax11.set_xlabel('Distance [m]')
    ax1.legend(loc='upper right')

#    plt.show()
    return fig, ax1, ax11

#def plot_3D_traces(traces,var = 'x'):
#
#    n_turns = traces.z.shape[0]
#    fig = plt.figure(figsize=(8, 10))
#    seq_col_brew = sns.color_palette("Blues_r", n_turns)
#    sns.set_palette(seq_col_brew)
#
#    ax1 = fig.add_subplot(211)
#    ax11 = ax1.twiny()
#
#
#    for i in xrange(n_turns):
#        z=traces.z[i,:]
#        t = z/c
#        d = getattr(traces,var)[i,:]
#        ax1.plot(t*1e9,d,'.')
#        ax11.plot(z, np.zeros(len(z)))
#        ax11.cla()
#
#    ax1.set_xlabel('Time [ns]')
#    ax1.set_ylabel('Signal [mm]')
#    ax11.set_xlabel('Distance [m]')
#
#    plt.show()


def plot_frequency_responses(data, labels, f_c, amp_range=(1e-2,4), phase_range=(-45.,45.)):
    fig = plt.figure(figsize=(10, 6))

    ax1 = fig.add_subplot(211)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    for i, (d, l) in enumerate(zip(data,labels)):
        ax1.plot(d[0],d[1],'-',label = l)

    ax1.set_xlabel('Signal frequency [Hz]')
    ax1.set_ylabel('V$_{out}$/V$_{in}$')
    ax1.legend(loc='lower left')
    ax1.set_xticklabels(())
    ax1.annotate('Cut-off\nfrequency', xy=(1.1*f_c, 1.2*amp_range[0]), xytext=(2*f_c, 3*amp_range[0]),
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="arc3"))
    ax1.annotate('-3dB', xy=(1.1*min(d[0]), 0.65), xytext=(3*min(d[0]), 2e-1),
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="arc3"))
    ax1.axhline(y=1/np.sqrt(2),c="black", ls='--')
    ax1.axvline(x=f_c,c="black", ls='--')
    ax1.set_ylim([amp_range[0], amp_range[1]])
    ax2 = fig.add_subplot(212)

    for i, (d, l) in enumerate(zip(data,labels)):
        map_valid =  d[1] > amp_range[0]
        ax2.plot(d[0][map_valid],d[2][map_valid],'-')

    ax2.set_xscale("log")
    ax2.axhline(y=-45.,c="black", ls='--')
    ax2.axhline(y=45.,c="black", ls='--')
    ax2.axvline(x=f_c,c="black", ls='--')
    ax2.set_xlabel('Signal frequency [Hz]')
    ax2.set_ylabel('Phase shift [deg]')
    ax2.set_yticks([-90, -67.5,-45,-22.5,0,22.5,45,67.5,90])
    ax2.set_ylim([-93, 93])
    ax2.title.set_visible(False)

    fig.subplots_adjust(hspace = .05)
    plt.show()
    return fig, ax1, ax2


def plot_debug_data(processors, source = 'input'):


    def pick_signals(processor, source = 'input'):
        """
        A function which helps to visualize the signals passing the signal processors.
        :param processor: a reference to the signal processor
        :param source: source of the signal, i.e, 'input' or 'output' signal of the processor
        :return: (t, z, bins, signal), where 't' and 'z' are time or position values for the signal values (which can be used
            as x values for plotting), 'bins' are data for visualizing sampling and 'signal' is the actual signal.
        """

        if source == 'input':
            bin_edges = processor.input_parameters['bin_edges']
            raw_signal = processor.input_signal
        elif source == 'output':
            bin_edges = processor.output_parameters['bin_edges']
            raw_signal = processor.output_signal
        else:
            raise ValueError('Unknown value for the data source')
        z = np.zeros(len(raw_signal)*4)
        bins = np.zeros(len(raw_signal)*4)
        signal = np.zeros(len(raw_signal)*4)
        value = 1.

        for i, edges in enumerate(bin_edges):
            z[4*i] = edges[0]
            z[4*i+1] = edges[0]
            z[4*i+2] = edges[1]
            z[4*i+3] = edges[1]
            bins[4*i] = 0.
            bins[4*i+1] = value
            bins[4*i+2] = value
            bins[4*i+3] = 0.
            signal[4*i] = 0.
            signal[4*i+1] = raw_signal[i]
            signal[4*i+2] = raw_signal[i]
            signal[4*i+3] = 0.
            value *= -1

        t = z/c
        return (t, z, bins, signal)

    fig = plt.figure(figsize=(8, 6))

    ax1 = fig.add_subplot(211)
    ax11 = ax1.twiny()
    ax2 = fig.add_subplot(212)
    ax22 = ax2.twiny()

    coeff = 1.


    for i, processor in enumerate(processors):
        if source == 'input':
            if hasattr(processor, 'input_signal'):
                if processor.debug:
                    t, z, bins, signal = pick_signals(processor,'input')
                    label=processor.label
                    ax1.plot(t*1e9,bins*coeff, label=label)
                    ax11.plot(z, np.zeros(len(z)))
                    ax11.cla()
                    coeff *= 0.9
                    ax2.plot(t*1e9,signal*1e3)
                    ax22.plot(z, np.zeros(len(z)))
                    ax22.cla()
        elif source == 'output':
            if hasattr(processor, 'output_signal'):
                if processor.debug:
                    t, z, bins, signal = pick_signals(processor,'output')
                    label=processor.label
                    ax1.plot(t*1e9,bins*coeff, label=label)
                    ax11.plot(z, np.zeros(len(z)))
                    ax11.cla()
                    coeff *= 0.9
                    ax2.plot(t*1e9,signal*1e3)
                    ax22.plot(z, np.zeros(len(z)))
                    ax22.cla()

    ax1.set_ylim(-1.1,1.1)
    ax1.set_xticklabels(())
    ax1.legend(loc='upper right')
    ax11.set_xlabel('Distance [m]')

    ax2.set_xlabel('Time [ns]')
    ax2.set_ylabel('Signal [mm]')
    ax22.set_xticklabels(())

    plt.show()
    return fig, ax1, ax2
