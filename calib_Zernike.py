import matplotlib.pyplot as plt
from glob import glob
import numpy as np
import cv2
from Pycaso import pattern
from Pycaso import data_library as data
from Pycaso import pycaso as pcs
from Pycaso import solve_library as solvel


def calib_Zernike (nZ : int,
                   date : str,
                   reverse : bool = False) :


  """ Calibration by Zernike method and 3D plotting of the calibration

  Args :
    nZ : int
        Polynomial degree of the Zernike polynome
    date : str
        Date of the test
    calibration_dict : dict
        Dictionnary ot the calibration, with the calibration folders, and the ChAruCo dimensions
    reverse : bool, optional
        Indicates if the right camera images should be reversed. 'True' for the first calibration only

  """

  saving_folder = f'./{date}/results_calib/nZ_{nZ}/'

  #Dictionnary ot the calibration, with the calibration folders, and the ChAruCo dimensions
  calibration_dict = {
    'cam1_folder' : f'./{date}/r',
    'cam2_folder' : f'./{date}/l',
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
    x3_list.append(20 +5*i)
  x3_list = np.array(x3_list)

  print('')
  print(date)
  print('#####       ')
  print('Zernike method - Start calibration')
  print('#####       ')

  #calibration : Zernike constants, magnification
  A_Zernike, Magnification = pcs.Zernike_calibration(z_list = x3_list,
                                                     Zernike_pform = nZ,
                                                     plotting = False,
                                                     iterations = 9,
                                                     **calibration_dict)

  coord = np.load(saving_folder+'3D_coordinates/3D_coordinates_Zernike.npy')

  #display the coordinates in the 3D space
  xc = []
  yc = []
  zc = []
  for i in range(len(coord[0])):
    xc.append(coord[0][i])
    yc.append(coord[1][i])
    zc.append(coord[2][i])
  ax = plt.figure().add_subplot(111,projection='3d')
  ax.scatter(xc,yc,zc)
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')
  plt.savefig(f'./{date}/results_calib/nZ_{nZ}/3D_coordinates/'+'Coord3D')
  plt.show()

  np.save(saving_folder+'A_Zernike.npy', A_Zernike)


if __name__ == "__main__":

    date = "2025_06_18"
    nZ = 11 #polynomial degree

    calib_Zernike(nZ, date, False)
