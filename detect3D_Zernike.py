import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import pathlib
from math import *
from Pycaso import solve_library as solvel
from Pycaso import data_library as data
from Pycaso import pycaso as pcs


def f(all_pxl, all_pyl, all_pxr, all_pyr):
  LA_allp = []
  for i in range(len(all_pyr)):
    A_allp = np.zeros((2,len(all_pyr[0]),2))
    for j in range(len(all_pyr[0])):
      A_allp[0][j][0] = all_pyl[i][j]
      A_allp[0][j][1] = all_pxl[i][j]
      A_allp[1][j][0] = all_pyr[i][j]
      A_allp[1][j][1] = all_pxr[i][j]
    LA_allp.append(A_allp)
  return LA_allp


def RtoL_transfo(rightpoints, matrix):
  Rightp = []
  for i in range(len(rightpoints)):
    vect = list(rightpoints[i])
    vect.append(1)
    vectp = np.dot(matrix, vect)
    vectp = list(vectp)
    vectp[0] = vectp[0]/vectp[2]
    vectp[1] = vectp[1]/vectp[2]
    del vectp[2]
    Rightp.append(vectp)
  return np.array(Rightp)


if __name__ == '__main__':

    date = '2025_05_15'
    sample = 'SC37_40_A1L'
    nZ = 12
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


    all_pxl = np.load(saving_folder + 'all_pxl.npy', allow_pickle=True)
    all_pyl = np.load(saving_folder + 'all_pyl.npy', allow_pickle=True)
    all_pxr = np.load(saving_folder + 'all_pxr.npy', allow_pickle=True)
    all_pyr = np.load(saving_folder + 'all_pyr.npy', allow_pickle=True)
    Lp = f(all_pxl, all_pyl, all_pxr, all_pyr)

    #for i in range(len(Lp)):
        #Lrp=RtoL_transfo(Lp[i][1],M)
        #Diff=[]
        #for j in range(len(Lrp)):
            #Diff.append(Lp[i][0][j][1] - Lrp[j][1])
        #diff=sum(Diff)/len(Diff)
        #for j in range(len(Lrp)):
            #Lrp[j][1]+=diff
        #Lp[i][1]=Lrp



#LA BONNE IDEE
    Lrp = RtoL_transfo(Lp[0][1], M)
    #Lfalse = []
    #for j in range(len(Lrp)):
##      if Lp[0][0][j][0] - Lrp[j][0] > 40 or Lp[0][0][j][0] - Lrp[j][0] < 20: #pour 9 degrés et 10 degrés l=9cm
##      if Lp[0][0][j][0] - Lrp[j][0] > 90 or Lp[0][0][j][0] - Lrp[j][0] < 70: #pour 18 degrés
##      if Lp[0][0][j][0] - Lrp[j][0] > 60 or Lp[0][0][j][0] - Lrp[j][0] < 40: #pour 20 degrés l=15 cm
##      if Lp[0][0][j][0] - Lrp[j][0] > 115 or Lp[0][0][j][0] - Lrp[j][0] < 90: #pour 28 degrés
##      if Lp[0][0][j][0] - Lrp[j][0] > 80 or Lp[0][0][j][0] - Lrp[j][0] < 60: #pour 30 degrés l=20 cm
##      if Lp[0][0][j][0] - Lrp[j][0] > 140 or Lp[0][0][j][0] - Lrp[j][0] < 120: #pour 40 degrés
      #if Lp[0][0][j][1] - Lrp[j][1] > 160 or Lp[0][0][j][1] - Lrp[j][1] < 130: #pour 40 degrés l=31 cm
##      if Lp[0][0][j][1] - Lrp[j][1] > 190 or Lp[0][0][j][1] - Lrp[j][1] < 140:
        #Lfalse.append([Lrp[j], j])
    #print(len(Lfalse))
    #Lid = []
    #for j in range(len(Lfalse)):
        #for k in range(len(Lfalse)):
##          if Lp[0][0][Lfalse[j][1]][0] - Lfalse[k][0][0] > 10 and Lp[0][0][Lfalse[j][1]][0] - Lfalse[k][0][0] < 50: #pour 9 degrés et 10 degrés l=9cm
##          if Lp[0][0][Lfalse[j][1]][0] - Lfalse[k][0][0] > 60 and Lp[0][0][Lfalse[j][1]][0] - Lfalse[k][0][0] < 100: #pour 18 degrés
##          if Lp[0][0][Lfalse[j][1]][0] - Lfalse[k][0][0] > 40 and Lp[0][0][Lfalse[j][1]][0] - Lfalse[k][0][0] < 60: #pour 20 degrés l=15 cm
##          if Lp[0][0][Lfalse[j][1]][0] - Lfalse[k][0][0] > 80 and Lp[0][0][Lfalse[j][1]][0] - Lfalse[k][0][0] < 120: #pour 28 degrés
##          if Lp[0][0][Lfalse[j][1]][0] - Lfalse[k][0][0] > 60 and Lp[0][0][Lfalse[j][1]][0] - Lfalse[k][0][0] < 90: #pour 30 degrés l=20 cm
##          if Lp[0][0][Lfalse[j][1]][0] - Lfalse[k][0][0] > 110 and Lp[0][0][Lfalse[j][1]][0] - Lfalse[k][0][0] < 150: #pour 40 degrés
          #if Lp[0][0][Lfalse[j][1]][1] - Lfalse[k][0][1] > 120 and Lp[0][0][Lfalse[j][1]][1] - Lfalse[k][0][1] < 180: #pour 40 degrés l=31 cm
##            if abs(Lp[0][0][Lfalse[j][1]][1] - Lfalse[k][0][1]) < 2: #pour 9 degrés et 20 degrés l=15 cm et 10 degrés l=9cm
##            if abs(Lp[0][0][Lfalse[j][1]][1] - Lfalse[k][0][1]) < 3: #pour 18 degrés
##            if abs(Lp[0][0][Lfalse[j][1]][1] - Lfalse[k][0][1]) < 4: #pour 28 degrés et 30 degrés l=20 cm
            #if abs(Lp[0][0][Lfalse[j][1]][1] - Lfalse[k][0][1]) < 7: #pour 40 degrés et pour 40 degrés l=31 cm
              #Lid.append([Lfalse[j][1], Lfalse[k][1]])
    #print(len(Lid))
    #for i in range(len(Lp)):
      #Rightbuff = Lp[i][1].copy()
      #for j in range(len(Lid)):
        #Lp[i][1][Lid[j][0]] = Rightbuff[Lid[j][1]]

    Lfalse = []
    Lid = []
    for j in range(len(Lrp)):
        if abs(Lp[0][0][j][0] - Lrp[j][0]) > 10:
            Lfalse.append([Lrp[j], j])
        print(Lp[0][0][j][0] - Lrp[j][0])
    print(len(Lfalse))
    print(Lfalse)

    for j in range(len(Lfalse)):
        for k in range(len(Lfalse)):
            if abs(Lp[0][0][Lfalse[j][1]][0] - Lfalse[k][0][0]) < 10 :
                if Lp[0][0][Lfalse[j][1]][1] - Lfalse[k][0][1]> 100 and Lp[0][0][Lfalse[j][1]][1] - Lfalse[k][0][1]<150:
                    Lid.append([Lfalse[j][1], Lfalse[k][1]])
    print(len(Lid))
    print(Lid)

    for i in range(len(Lp)):
        Rightbuff = Lp[i][1].copy()
        for j in range(len(Lid)):
            Lp[i][1][Lid[j][0]] = Rightbuff[Lid[j][1]]



    Lx3d = []
    Ly3d = []
    Lz3d = []

    for i in range(len(Lp)):
        Left, Right= Lp[i]
        Z_solution = pcs.Zernike_identification(Left,
                                                Right,
                                                A_constant,
                                                nZ,
                                                C_dim)
        x,y,z = Z_solution
        #x=np.flip(x)
        #y=np.flip(y)
        #z=np.flip(z)
        Lx3d.append(x)
        Ly3d.append(y)
        Lz3d.append(z)
    print(x,y,z)

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

