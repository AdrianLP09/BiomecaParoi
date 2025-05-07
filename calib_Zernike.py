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

    date = "2025_05_05"

    nZ = 12 #polynomial degree

    saving_folder = f'./{date}/results_calib/nZ_{nZ}/'


    # Define the inputs
    calibration_dict = {
      'cam1_folder' : f'./data/SC37_40_4DFIXNR/left_12x12_5',
      'cam2_folder' : f'./data/SC37_40_4DFIXNR/right_12x12_5',
      'name' : 'calibration',
      'saving_folder' : saving_folder,
      'ncx' : 12,
      'ncy' : 12,
      'sqr' : 7.5}  #in mm

    DIC_dict={
      'cam1_folder':f'./data/SC37_40_4DFIXNR/left_12x12_5',
      'cam2_folder':f'./data/SC37_40_4DFIXNR/right_12x12_5',
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
    print('Zernike method - Start calibration')
    print('#####       ')

    ##reverse the right images, cameras are in mirror
    #Liste_image  = sorted(glob(f'./{date}/r/'+"0*"))
    #for image in Liste_image:
      #img=cv2.imread(image)
      #img=cv2.rotate(img,cv2.ROTATE_180)
      #cv2.imwrite(image,img)




    #calibration : Zernike constants, magnification, points detected on the ChAruco, theoretical points, number of points and 3D points
    A_Zernike, Magnification = pcs.Zernike_calibration(z_list = x3_list,
                                                       Zernike_pform = nZ,
                                                       plotting = False,
                                                       iterations = 6,
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



    #try:
      #z_list=np.load(f'./{date}/nZ_{nZ}/results_calib/Pycaso_retroprojection_error/z_list_Zernike.npy')
      #Xf=np.zeros((n,X.shape[1],3))
      #xthf=np.zeros((n,X.shape[1],3))

      #for i in range(n//2):
        #for j in range(len(X[0])):
          #Xf[i][j]=np.insert(X[i][j],2,z_list[i][j])
          #Xf[i+n//2][j]=np.insert(X[i+n//2][j],2,z_list[i][j])
          #xthf[i][j]=np.insert(xth[i][j],2,z_list[i][j])
          #xthf[i+n//2][j]=np.insert(xth[i+n//2][j],2,z_list[i][j])

          #print(Xf)
          #plt.imshow(Xf)


    #except:
      #print('No retroprojection')

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

    ##Z_mean,Z_std,Z_iter = pcs.retroprojection_error('Zernike',
                                                     ##nZ,
                                                     ##A_Zernike,
                                                     ##z_points,
                                                     ##X,
                                                     ##xth,
                                                     ##cam1_folder=f'./{date}/r',
                                                     ##cam2_folder=f'./{date}/l',
                                                     ##ncx=12,
                                                     ##ncy=12,
                                                     ##sqr=7.5)



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


    #ver = crappy.blocks.VideoExtenso(camera='XiAPI',
                                   #config=True,
                                   #save_images=True,
                                   #save_folder=f'./{date}/40d_cd/right/',
                                   #labels=['tr(s)', 'meta_r', 'pix_r', 'eyy_r', 'exx_r'],
                                   #white_spots=False,
                                   #**{"serial_number": "14482450",
                                      #"exposure": 50000,
                                      #"trigger": "Hdw after config"})

    #vel = crappy.blocks.VideoExtenso(camera='XiAPI',
                                   #config=True,
                                   #save_images=True,
                                   #save_folder=f'./{date}/40d_cd/left/',
                                   #labels=['tl(s)', 'meta_l', 'pix_l', 'eyy_l', 'exx_l'],
                                   #white_spots=False,
                                   #**{"serial_number": "32482550",
                                      #"exposure": 50000,
                                      #"trigger": "Hdw after config"})


    #ftdi = crappy.blocks.IOBlock('Ft232r', cmd_labels=['cmd'], spam=False, direction=0b0001, URL='ftdi://ftdi:232:A105QJ01/1')

    #flow_ali = crappy.blocks.IOBlock('Flow_controller_alicat',
                                   #port="/dev/ttyUSB0",
                                   #startbit=1,
                                   #databits=8,
                                   #parity="N",
                                   #stopbits=1,
                                   #errorcheck="crc",
                                   #baudrate=9600,
                                   #method="RTU",
                                   #timeout=3,
                                   #svp=['Pressure', 'Mass_flow'],
                                   #cmd_labels=['flowcmd'],
                                   #labels=['t(s)', 'press', 'mass_flow'])

    #trid = PolyZernike.ControlZernike(Zernikecoeffs = A_Zernike,
                               #matrix_file = f'./{date}/transfomatrix.npy',
                               #label = 'triexx',
                               #pform = nZ)


    #trigcam = triggerflow.Triggercamexx(cmd_labels=['triexx', 'exxcmd'])

    #gen_exx = crappy.blocks.Generator([{'type': 'Constant',
                                      #'value': 0.15,
                                      #'condition': 'path_id>1'}], cmd_label='exxcmd', spam=True)

    #gen_dexx = crappy.blocks.Generator([{'type': 'Constant',
                                     #'value': 0.005,
                                     #'condition':'delay=30'}], cmd_label='dexxcmd', spam=True)

    #p = 4400
    #i = 1800
    #d = 0

    #pid = crappy.blocks.PID(kp=p,
                          #ki=i,
                          #kd=d,
                          #out_max=100,
                          #out_min=0,
                          #setpoint_label='dexxcmd',
                          #input_label='dexx',
                          #labels=['t(s)', 'flowcmd'])

    #rec_tridexx = crappy.blocks.Recorder(file_name = './2023_08_22/40d_cd/defexxsvp')

    #rec_pid = crappy.blocks.Recorder(file_name='./2023_08_22/40d_cd/pid')

    #rec_ali = crappy.blocks.Recorder(file_name='./2023_08_22/40d_cd/ali')


    #crappy.link(gen_exx, trigcam)
    #crappy.link(trid, trigcam)
    #crappy.link(trigcam, ftdi)
    #crappy.link(trigcam, gen_exx)

    #crappy.link(vel, trid)
    #crappy.link(ver, trid)

    #crappy.link(gen_dexx, pid)
    #crappy.link(trid, pid)
    #crappy.link(pid, flow_ali)

    #crappy.link(trid, rec_tridexx)
    #crappy.link(pid, rec_pid)
    #crappy.link(flow_ali, rec_ali)

    #crappy.start()
