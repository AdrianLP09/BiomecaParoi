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


if __name__ == '__main__' :  

    date = '2025_06_02'
    sample = 'SC37_20'
    spform = 333
    data_folder = f'./{date}/results_calib/Spform_{spform}/'
    saving_folder = f'./{date}/{sample}/'


    if os.path.exists(saving_folder+f'Spform_{spform}/') :
        ()
    else :
        P = pathlib.Path(saving_folder+f'Spform_{spform}/')
        pathlib.Path.mkdir(P, parents = True)

    S_constants0 = np.load(data_folder+'S_constants0.npy')
    S_constants = np.load(data_folder+'S_constants.npy')

    M = np.load(f'./{date}/{sample}/transfomatrix.npy')

    
    Lx3d = []
    Ly3d = []
    Lz3d = []
    Lp = np.load(saving_folder + 'Lp.npy',allow_pickle=True)
    for i in range(len(Lp)):
      Left, Right = Lp[i]
      xSoloff_solution = pcs.Soloff_identification (Left,
                                                    Right,
                                                    S_constants0,
                                                    S_constants,
                                                    Soloff_pform = spform,
                                                    method = 'curve_fit')
      x,y,z = xSoloff_solution
      Lx3d.append(x)
      Ly3d.append(y)
      Lz3d.append(z)

    np.savetxt(saving_folder + f'Spform_{spform}/X3d.txt', Lx3d)
    np.savetxt(saving_folder + f'Spform_{spform}/Y3d.txt', Ly3d)
    np.savetxt(saving_folder + f'Spform_{spform}/Z3d.txt', Lz3d)


    fig=plt.figure(figsize=(16,9))
    ax=plt.axes(projection='3d')
    ax.grid(visible=True,
            color='grey',
            linestyle='-.',
            linewidth=0.3,
            alpha=0.2)
    my_cmap=plt.get_cmap('hsv')
    sctt=ax.scatter3D(x,y,z, alpha=0.8, c=z, cmap=my_cmap)
    plt.title('Results')
    fig.colorbar(sctt, ax=ax, shrink=0.5, aspect=5)
    plt.show()
