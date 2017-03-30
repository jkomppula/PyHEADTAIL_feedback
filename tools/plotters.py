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
