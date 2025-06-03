import crappy
import matplotlib.pyplot as plt
import ft232R #classe InOut pour le Ft232r
import triggerflow # classes de trigger de l'expérience
#import PolyZernike
from glob import glob
import numpy as np
import cv2
from Pycaso import pattern
from Pycaso import data_library as data
from Pycaso import pycaso as pcs
from Pycaso import solve_library as solvel



if __name__ == "__main__":

    date = "2025_06_02"

    l_pform = 4   #polynomial degree

    saving_folder = f'./{date}/results_calib/Lpform_{l_pform}/'


    # Define the inputs
    calibration_dict = {
      'cam1_folder' : f'./{date}/l',
      'cam2_folder' : f'./{date}/r',
      'name' : 'calibration',
      'saving_folder' : saving_folder,
      'ncx' : 12,
      'ncy' : 12,
      'sqr' : 7.5}  #in mm


    DIC_dict={
      'cam1_folder' : f'./{date}/l',
      'cam2_folder' : f'./{date}/r',
      'saving_folder': saving_folder,
      'window':[[300,1700],[300,1700]]}


    # Create the list of z plans
    x3_list = []
    for i in range(21) :
      x3_list.append(120 -5*i)
    x3_list=np.array(x3_list)

    print('')
    print(date)
    print('#####       ')
    print('Lagrange method - Start calibration')
    print('#####       ')




##ROTATE_180 dans le cas où les images des deux caméras sont inversées.
    #Liste_image  = sorted(glob(f'./{date}/r/'+"0*")) #list of image
    #for image in Liste_image:
      #img=cv2.imread(image)
      #img=cv2.rotate(img,cv2.ROTATE_180)
      #cv2.imwrite(image,img)

    L_constants, Mag = pcs.Lagrange_calibration(z_list = x3_list,
                                                Lagrange_pform = l_pform,
                                                plotting = False,
                                                iterations = 10,
                                                **calibration_dict)

    np.save(saving_folder+'L_constants.npy', L_constants)

    coord=np.load(saving_folder+'3D_coordinates/3D_coordinates_Lagrange.npy')

    #display the coordinates in the 3D space

    xc=[]
    yc=[]
    zc=[]
    for i in range(len(coord[0])):
      xc.append(coord[0][i])
      yc.append(coord[1][i])
      zc.append(coord[2][i])
    ax=plt.figure().add_subplot(111,projection='3d')
    ax.scatter(xc,yc,zc)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


    #X1,X2 = data.DIC_get_positions(DIC_dict)

    #Lagrange_results = pcs.Lagrange_identification(X1[0],
                                                   #X2[0],
                                                   #L_constants,
                                                   #l_pform)

    #np.save(f'./{date}/results_calib/Lpform_{l_pform}/Lagrange_results.npy', Lagrange_results)
    #print(Lagrange_results)

