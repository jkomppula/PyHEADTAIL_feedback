import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection

def plot3Dmultidamping(tracker, var):

    data =  getattr(tracker,var)

    n_turns = len(data)
    n_points = len(data[0])
    z = tracker.z[0]
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

    surf = ax.plot_wireframe(z, turns, plot_data)
    ax.invert_yaxis()
    ax.set_xlabel('Z [m]')
    ax.set_ylabel('Turn')
    ax.set_zlabel('Variable: ' + var)
    plt.show()
    return fig, ax

def plot_frequency_responses(data, labels, f_c):
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
    ax1.annotate('Cut-off\nfrequency', xy=(1.1*f_c, 1.2e-2), xytext=(2*f_c, 3e-2),
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="arc3"))
    ax1.annotate('-3dB', xy=(1.1*min(d[0]), 0.65), xytext=(3*min(d[0]), 2e-1),
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="arc3"))
    ax1.axhline(y=1/np.sqrt(2),c="black", ls='--')
    ax1.axvline(x=f_c,c="black", ls='--')
    ax1.set_ylim([1e-2, 4])
    ax2 = fig.add_subplot(212)

    for i, (d, l) in enumerate(zip(data,labels)):
        ax2.plot(d[0],d[2],'-')

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
    return fig, ax