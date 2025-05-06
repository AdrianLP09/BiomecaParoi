import numpy as np
import sigfig as sgf
import sys
import pathlib
import os
import solve_library_direct as solvel 
import data_library_direct as data
from math import *


def magnification (X1, X2, x1, x2) :
    """Calculation of the magnification between reals and detected positions
    
    Args:
       X1 : numpy.ndarrayx
           Organised detected positions (X1 = X axe)
       X2 : numpy.ndarray
           Organised detected positions (X2 = Y axe)
       x1 : numpy.ndarray
           Organised real positions (x1 = x axe)
       x2 : numpy.ndarray
           Organised real positions (x2 = y axe)
    Returns:
       Magnification : int
           Magnification between detected and real positions
           [Mag x, Mag y]
    """
    Delta_X1 = np.nanmean(abs(X1-np.nanmean(X1)))
    Delta_X2 = np.nanmean(abs(X2-np.nanmean(X2)))
    Delta_x1 = np.nanmean(abs(x1-np.nanmean(x1)))
    Delta_x2 = np.nanmean(abs(x2-np.nanmean(x2)))
    Magnification = np.asarray([Delta_x1/Delta_X1, Delta_x2/Delta_X2]) 
    return (Magnification)




def direct_calibration (__calibration_dict__,
             x3_list,
             saving_folder,
             direct_polynome_degree,
             detection = True) :
    """Calculation of the magnification between reals and detected positions 
    and the calibration parameters A:--> x = A.M(X)
    
    Args:
       __calibration_dict__ : dict
           Calibration properties define in a dict. Including 'left_folder', 
           'right_folder', 'name', 'ncx', 'ncy', 'sqr'
       x3_list : numpy.ndarray
           List of the different z position. (WARNING : Should be order the 
                                              same way in the target folder)
       saving_folder : str
           Folder to save datas
       direct_polynome_degree : int, optional
           Polynomial degree
       detection : bool, optional
           If True, all the analysis will be done. If False, the code will 
           take the informations in 'saving_folder'

    Returns:
       A : numpy.ndarray
           Constants of direct polynomial
       Magnification : int
           Magnification between reals and detected positions
    """
    
    if direct_polynome_degree == 1 :
        direct_A = np.zeros((3, 5))
    elif direct_polynome_degree == 2 :
        direct_A = np.zeros((3, 15))
    elif direct_polynome_degree == 3 :
        direct_A = np.zeros((3, 35))
    elif direct_polynome_degree == 4 :
        direct_A = np.zeros((3, 70))
    else :
        print ('Only define for polynomial degrees (1, 2, 3 or 4')
        sys.exit()
    # Detect points from folders
    all_Ucam, all_Xref, nb_pts = data.pattern_detection(__calibration_dict__, 
                                                        detection = detection, 
                                                        saving_folder = saving_folder)        

    # Creation of the reference matrix Xref and the real position Ucam for 
    # each camera i
    x, Xc1, Xc2 = data.camera_np_coordinates(all_Ucam, all_Xref, x3_list)

    # Plot the references plans
    solvel.refplans(x, x3_list)

    # Calcul of the Soloff polynome's constants. X = A . M
    Magnification = np.zeros((2, 2))

    for camera in [1, 2] :
        if camera == 1 :
            X = Xc1
        elif camera == 2 :
            X = Xc2
        x1, x2, x3 = x
        X1, X2 = X
        
        # Compute the magnification (same for each cam as set up is symetric)
        Magnification[camera-1] = magnification (X1, X2, x1, x2)
        
        # Do the system x = Ap*M, where M is the monomial of the real 
        # coordinates of crosses and x the image coordinates, and M the unknow
        M = solvel.Direct_Polynome({'polynomial_form' : direct_polynome_degree}).pol_form(Xc1, Xc2)
        Ap = np.matmul(x, np.linalg.pinv(M))
        direct_A = Ap

        # Error of projection
        xd = np.matmul(Ap,M)
        proj_error = x - xd
        print('Max ; min projection error (polynomial form ',
              str(direct_polynome_degree),
              ') for camera ',
              str(camera),
              ' = ',
              str(sgf.round(np.amax(proj_error), sigfigs = 3)),
              ' ; ',
              str(sgf.round(np.amin(proj_error), sigfigs = 3)),
              ' px')
    return(direct_A, Magnification)    



if __name__ == '__main__' :  

    date = "2025_04_09"
  
    # Define the inputs
    __calibration_dict__ = {
    'left_folder' : f'./{date}/l',
    'right_folder' : f'./{date}/r',
    'name' : 'micro_calibration',
    'ncx' : 12,
    'ncy' : 12,
    'sqr' : 7.5}  #in mm

    # Create the list of z plans   
    x3_list = []
    for i in range(21) :
      x3_list.append(20 + 5*i)  

    # Chose the polynomial degree for the calibration fitting
    direct_polynome_degree = 4

    saving_folder_direct = f'./{date}/results_calib_direct/'
    
    # Create the result folder if not exist
    if os.path.exists(saving_folder_direct) :
      ()
    else :
      P = pathlib.Path(saving_folder_direct)
      pathlib.Path.mkdir(P, parents = True)

    print('')
    print('#####       ')
    print('Start calibration')
    print('#####       ')

    direct_A, Magnification = direct_calibration (__calibration_dict__,
                                                 x3_list,
                                                 saving_folder_direct,
                                                 direct_polynome_degree = direct_polynome_degree,
                                                 detection = True)
                                                 
    np.save(f'./{date}/coeff_direct', direct_A)
    
