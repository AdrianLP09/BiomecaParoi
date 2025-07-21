import matplotlib.pyplot as plt
from glob import glob
import numpy as np
import cv2
from Pycaso import pattern
from Pycaso import data_library as data
from Pycaso import pycaso as pcs
from Pycaso import solve_library as solvel


def calib_Lagrange (l_pform : int,
                    date : str,
                    reverse : bool = False) :


    """ Calibration by Lagrange method and 3D plotting of the calibration

    Args :
      l_pform : int
          Polynomial degree of the Lagrange polynome
      date : str
          Date of the test
      reverse : bool, optional
          Indicates if the right camera images should be reversed. 'True' for the first calibration only

    """

    saving_folder = f'./{date}/results_calib/Lpform_{l_pform}/'

    #Dictionnary ot the calibration, with the calibration folders, and the ChAruCo dimensions
    calibration_dict = {
    'cam1_folder' : f'./{date}/l',
    'cam2_folder' : f'./{date}/r',
    'name' : 'calibration',
    'saving_folder' : saving_folder,
    'ncx' : 12,
    'ncy' : 12,
    'sqr' : 7.5}  #in mm

    if reverse :
      #reverse the right images, cameras are in mirror
      Liste_image  = sorted(glob(f'./{date}/r/'+"0*"))
      for image in Liste_image:
        img = cv2.imread(image)
        img = cv2.rotate(img,cv2.ROTATE_180)
        cv2.imwrite(image,img)

    # Create the list of z plans
    x3_list = []
    for i in range(21) :
      x3_list.append(120 -5*i)
    x3_list = np.array(x3_list)

    print('')
    print(date)
    print('#####       ')
    print('Lagrange method - Start calibration')
    print('#####       ')

    #calibration : Lagrange constants, magnification
    L_constants, Mag = pcs.Lagrange_calibration(z_list = x3_list,
                                                Lagrange_pform = l_pform,
                                                plotting = False,
                                                iterations = 10,
                                                **calibration_dict)

    coord=np.load(saving_folder+'3D_coordinates/3D_coordinates_Lagrange.npy')

    #display the coordinates in the 3D space
    xc=[]
    yc=[]
    zc=[]
    for i in range(len(coord[0])):
      xc.append(coord[0][i])
      yc.append(coord[1][i])
      zc.append(coord[2][i])
    ax = plt.figure().add_subplot(111,projection='3d')
    ax.scatter(xc,yc,zc)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.savefig(f'./{date}/results_calib/Lpform_{l_pform}/3D_coordinates/'+'Coord3D')
    plt.show()

    np.save(saving_folder+'L_constants.npy', L_constants)


if __name__ == "__main__":

    date = "2025_06_18"
    l_pform = 4   #polynomial degree

    # Define the inputs


    calib_Lagrange(l_pform, date, False)
