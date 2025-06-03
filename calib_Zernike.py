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

    nZ = 9 #polynomial degree

    saving_folder = f'./{date}/results_calib/nZ_{nZ}/'


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
    print('Zernike method - Start calibration')
    print('#####       ')

    #reverse the right images, cameras are in mirror
    #Liste_image  = sorted(glob(f'./{date}/r/'+"0*"))
    #for image in Liste_image:
      #img=cv2.imread(image)
      #img=cv2.rotate(img,cv2.ROTATE_180)
      #cv2.imwrite(image,img)




    #calibration : Zernike constants, magnification, points detected on the ChAruco, theoretical points, number of points and 3D points
    A_Zernike, Magnification = pcs.Zernike_calibration(z_list = x3_list,
                                                       Zernike_pform = nZ,
                                                       plotting = False,
                                                       iterations = 10,
                                                       **calibration_dict)


    np.save(saving_folder+'A_Zernike.npy', A_Zernike)

    xth=np.load(saving_folder+'all_xth_calibration.npy')
    X=np.load(saving_folder+'all_X_calibration.npy')
    coord=np.load(saving_folder+'3D_coordinates/3D_coordinates_Zernike.npy')
    print(np.load(saving_folder+'nb_pts_calibration.npy'))
    n=X.shape[0]


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

    #Xff=np.zeros((n,3,X.shape[1]))
    #for i in range(n):
        #for j in range(len(Xf[0])):
                          #Xff[i][0][j]=Xf[i][j][0]
                          #Xff[i][1][j]=Xf[i][j][1]
                          #Xff[i][2][j]=Xf[i][j][2]
    #for i in range(len(Xff)):
            #Xff[i]=np.nan_to_num(Xff[i],
                                       #nan=np.nanmean(Xff[i])) #enlève les possibles NaN


    #fit,errors,mean_error,residual=solvel.fit_plans_to_points(Xff)
    #np.save(f'./{date}/nZ_{nZ}/results_calib/fit.npy',fit)
    #np.save(f'./{date}/nZ_{nZ}/results_calib/errors.npy',errors)
    #np.save(f'./{date}/nZ_{nZ}/results_calib/mean_error.npy',mean_error)
    #np.save(f'./{date}/nZ_{nZ}/results_calib/residual.npy',residual)



    #z_points=np.ones((n//2,np.shape(X)[1]))
    #for i in range(np.shape(X)[0]//2):
      #z_points[i]=z_points[i]*x3_list[i]

    #A_Zernike=np.load(f'./{date}/nZ_{nZ}/results_calib/A_Zernike.npy')


    #get the points of cam1 and cam2
    #X1,X2 = data.DIC_get_positions(DIC_dict)

    #size=data.cameras_size(**calibration_dict)


    ##Identification in the 3D space of the detected points
    #Zernike_results = pcs.Zernike_identification(Xc1_identified = X1[0],
                                                 #Xc2_identified = X2[0],
                                                 #Zernike_constants = A_Zernike,
                                                 #Zernike_pform = nZ,
                                                 #Cameras_dimensions = size)

    #np.save(f'./{date}/nZ_{nZ}/results_calib/Zernike_results.npy', Zernike_results)



