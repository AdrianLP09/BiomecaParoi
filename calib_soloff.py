import crappy
import pycaso as pcs
import pattern
import ft232R #classe InOut pour le Ft232r
import triggerflow # classes de trigger de l'expérience
import PolyZernike
from glob import glob
import numpy as np
import cv2
import data_library as data

if __name__ == "__main__":

    date = "2025_04_28"

    Soloff_pform = 332   #polynomial degree

    saving_folder = f'./{date}/results_calib/Soloff_pform{Soloff_pform}/'


    # Define the inputs
    calibration_dict = {
      'cam1_folder' : f'./{date}/r',
      'cam2_folder' : f'./{date}/l',
      'name' : 'calibration',
      'saving_folder' : saving_folder,
      'ncx' : 12,
      'ncy' : 12,
      'sqr' : 7.5}  #in mm

    DIC_dict={
      'cam1_folder':f'./{date}/r',
      'cam2_folder':f'./{date}/l',
      'name':'identification',
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
    print('Soloff method - Start calibration')
    print('#####       ')




##ROTATE_180 dans le cas où les images des deux caméras sont inversées.
    #Liste_image  = sorted(glob(f'./{date}/r/'+"0*")) #list of image
    #for image in Liste_image:
      #img=cv2.imread(image)
      #img=cv2.rotate(img,cv2.ROTATE_180)
      #cv2.imwrite(image,img)

    S_constants0, S_constants, Mag = pcs.Soloff_calibration(z_list = x3_list,
                                                            Soloff_pform = Soloff_pform,
                                                            plotting = False,
                                                            iterations = 1)

    np.save(saving_folder+'S_constants0.npy', S_constants0)
    np.save(saving_folder+'S_constants.npy', S_constants)

    #X1,X2 = data.DIC_get_positions(DIC_dict)

    #Soloff_results = pcs.Soloff_identification(X1[0],
                                               #X2[0],
                                               #S_constants0,
                                               #S_constants,
                                               #Soloff_pform
                                               #method='curve_fit')

    #np.save(f'./{date}/results_calib/Soloff_pform{Soloff_pform}/Soloff_results.npy', Soloff_results)
    #print(Soloff_results)

