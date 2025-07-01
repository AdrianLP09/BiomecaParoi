import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import cv2
import os
import pathlib
from math import *
from Pycaso import solve_library as solvel
from Pycaso import data_library as data
from Pycaso import pycaso as pcs


def detect3D_Lagrange (l_pform : int,
                       date : str,
                       sample : str,
                       calibration_dict : dict
                       P_graph : int = 0) -> (list,
                                              list,
                                              list):

    """Identification of the points detected on both cameras cam1 and cam2
    into the global 3D-space using Lagrange method

    Args:
        l_pform : int
           Polynomial degree of the Lagrange polynome
        date : str
           Date of the test
        sample : str
           Type of sample (silicone + mesh + resorption)
        P_graph : int,optional
            State of the sample during the test which will be plotted

    Returns:
        Lx3d : list
            X Coordinates of the points in the 3d space, for each step of the test
        Ly3d : list
            Y Coordinates of the points in the 3d space, for each step of the test
        Lz3d : list
            Z Coordinates of the points in the 3d space, for each step of the test
    """

    data_folder = f'./{date}/results_calib/Lpform_{l_pform}/'
    saving_folder=f'./{date}/{sample}/'

    if os.path.exists(saving_folder+f'Lpform_{l_pform}/') :
        ()
    else :
        P = pathlib.Path(saving_folder+f'Lpform_{l_pform}/')
        pathlib.Path.mkdir(P, parents = True)

    L_constants = np.load(data_folder + 'L_constants.npy')


    Lx3d = []
    Ly3d = []
    Lz3d = []
    Lp = np.load(saving_folder + 'Lp.npy',allow_pickle=True)
    for i in range(len(Lp)):
        Left,Right= Lp[i]
        L_solution = pcs.Lagrange_identification (Left,
                                                  Right,
                                                  L_constants,
                                                  l_pform)

        x,y,z = L_solution
        Lx3d.append(x)
        Ly3d.append(y)
        Lz3d.append(z)


    x = Lx3d[P_graph]
    y = Ly3d[P_graph]
    z = Lz3d[P_graph]-Lz3d[0]
    fig = plt.figure(figsize=(16,9))
    ax = plt.axes(projection='3d')
    ax.grid(visible=True,
            color='grey',
            linestyle='-.',
            linewidth=0.3,
            alpha=0.2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_zlim(min(Lz3d[0]-Lz3d[0]), max(Lz3d[0]-Lz3d[P_graph]))
    #print(min(Lz3d[0]),max(Lz3d[-1]))
    my_cmap = plt.get_cmap('hsv')
    sctt = ax.scatter3D(x,y,z, alpha=0.8, c=z, cmap=my_cmap)
    plt.title('Results')
    fig.colorbar(sctt, ax=ax, shrink=0.5, aspect=5)
    plt.savefig(saving_folder + f'Lpform_{l_pform}/'+f'Figure_1_{P_graph}-{len(Lx3d)}')
    plt.show()
    np.savetxt(saving_folder+f'Lpform_{l_pform}/X3d.txt', Lx3d)
    np.savetxt(saving_folder+f'Lpform_{l_pform}/Y3d.txt', Ly3d)
    np.savetxt(saving_folder+f'Lpform_{l_pform}/Z3d.txt', Lz3d)

    return Lx3d, Ly3d, Lz3d


if __name__ == '__main__' :

    date = '2025_06_16'
    sample = "SC37_40_P7_16j"
    l_pform =4

    Lx3d, Ly3d, Lz3d = detect3D_Lagrange(l_pform, date, sample, 0)

