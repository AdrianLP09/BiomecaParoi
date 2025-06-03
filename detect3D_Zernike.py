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
    nZ = 5
    data_folder = f'./{date}/results_calib/nZ_{nZ}/'

    calibration_dict = {
      'cam1_folder' : f'./{date}/{sample}/video_extenso_left/',
      'cam2_folder' : f'./{date}/{sample}/video_extenso_right/',
      'name' : 'calibration',
      'saving_folder' : data_folder,
      'ncx' : 12,
      'ncy' : 12,
      'sqr' : 7.5}  #in mm

    saving_folder=f'./{date}/{sample}/'

    if os.path.exists(saving_folder+f'nZ_{nZ}/') :
        ()
    else :
        P = pathlib.Path(saving_folder+f'nZ_{nZ}/')
        pathlib.Path.mkdir(P, parents = True)


    M = np.load(f'./{date}/{sample}/transfomatrix.npy')

    A_constant = np.load(data_folder+'A_Zernike.npy')

    C_dim = data.cameras_size(**calibration_dict)


    Lx3d = []
    Ly3d = []
    Lz3d = []
    Lp = np.load(saving_folder + 'Lp.npy',allow_pickle=True)
    for i in range(len(Lp)):
        Left, Right= Lp[i]
        Z_solution = pcs.Zernike_identification(Left,
                                                Right,
                                                A_constant,
                                                nZ,
                                                C_dim)
        x,y,z = Z_solution
        Lx3d.append(x)
        Ly3d.append(y)
        Lz3d.append(z)

    np.savetxt(saving_folder + f'nZ_{nZ}/X3d.txt', Lx3d)
    np.savetxt(saving_folder + f'nZ_{nZ}/Y3d.txt', Ly3d)
    np.savetxt(saving_folder + f'nZ_{nZ}/Z3d.txt', Lz3d)

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
