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


if __name__ == '__main__' :  

    date = "2025_05_15"
    sample = 'SC37_40_A1L'
    spform = 222
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


    all_pxl = np.load(saving_folder + 'all_pxl.npy', allow_pickle=True)
    all_pyl = np.load(saving_folder + 'all_pyl.npy', allow_pickle=True)
    all_pxr = np.load(saving_folder + 'all_pxr.npy', allow_pickle=True)
    all_pyr = np.load(saving_folder + 'all_pyr.npy', allow_pickle=True)
    Lp = f(all_pxl, all_pyl, all_pxr, all_pyr)
    N = len(Lp[0][0])

### Appariement des points
    Template = cv2.imread(saving_folder+'Template_L.tiff') #Template Gimp
    Y_template,X_template = 904,863 #Constantes à modifier en fonction des coordonnées du Template

    Img = cv2.imread(saving_folder + 'Template_R.tiff') #Image 0 de la caméra de droite

    #On fait correspondre le template sur l'image de droite et on observe le décalage entre les deux images
    result = cv2.matchTemplate(Img, Template, cv2.TM_CCOEFF)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)
    (startX,startY) = maxLoc
    CoordTemplate = (Y_template, X_template, Y_template+Template.shape[0], X_template+Template.shape[1])
    Diff = (CoordTemplate[0]-maxLoc[0],CoordTemplate[1]-maxLoc[1])


    xr,yr = [],[]
    for i in range(len(Lp[0][0])):
        xr.append(Lp[0][1][i][1])
        yr.append(Lp[0][1][i][0])

    # On applique le delta aux coordonnées de droite
    xr2,yr2 = [],[]
    for i in range(N):
        yr2.append(yr[i] + Diff[0])
        xr2.append(xr[i] + Diff[1])

    Matched_Left = Lp[0][0]
    Matched_Right = np.array([(yr2[i],xr2[i]) for i in range(N)])
    row_ind = np.array([k for k in range(N)])
    col_ind = np.zeros(N,dtype=int)
    for i in range(N): # On fait l'appariement en regardant les points les plus proches après compensation du delta
      m = np.linalg.norm(Matched_Left[i]-Matched_Right[0])
      Id = 0
      for j in range(N):
        if np.linalg.norm(Matched_Left[i]-Matched_Right[j]) < m :
          #if np.all(np.not_equal(col_ind,j)):
          m = np.linalg.norm(Matched_Left[i]-Matched_Right[j])
          Id = j
          #else:
            #if np.linalg.norm(Matched_Left[i]-Matched_Right[j])< np.linalg.norm(Matched_Left[i]-Matched_Right[col_ind==j]):
              #m = np.linalg.norm(Matched_Left[i]-Matched_Right[j])
              #Id = j
      col_ind[i] = Id
    col_ind = np.array(col_ind)
    print(col_ind,len(col_ind))
    #row_ind, col_ind = linear_sum_assignment(distance_matrix)
    matched_gauche = Matched_Left[row_ind]
    matched_droite = Matched_Right[col_ind]
    for i in range(N):
        print(matched_gauche[i][1]-matched_droite[i][1])

    # Appariement
    for i in range(len(Lp)):
        Rightbuff = Lp[i][1].copy()
        for j in range(len(col_ind)):
            Lp[i][1][row_ind[j]] = Rightbuff[col_ind[j]]

    
    Lx3d = []
    Ly3d = []
    Lz3d = []
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
