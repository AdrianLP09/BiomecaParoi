import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from skimage.filters import threshold_otsu, difference_of_gaussians
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border
from skimage.util import invert
from glob import glob
import cv2
import math
import solve_library_soloff as solvel 
from math import *
from Pycaso import pycaso as pcs


############################################ DETECTION ANNIE ############################################
############################### Parameters
def return_num(s: str):
  try:
    num = int((s.split('/')[-1]).split()[0]) 
    return num
  except:
    num = 99999
    return num
    
def CoordCam(path=str, mask=str, savefile=str):
#  Liste_image  = sorted(glob(path+"*.tiff"), key=return_num) #list of image
#  if len(Liste_image)>1:
#    Liste_image = Liste_image[:15]
#  else:
#    print('PAS DE MASK')
#    return
  Liste_image  = sorted(glob(path+"0*")) #list of image
#  Liste_image = Liste_image[::7]
  Mask = path + mask # mask defined from the first image
  First_img = 1
  Last_img = 2000
  impair_coeff = 1. # impairs the thresh level for global binarization of spots
  Min_area = 10 #used to delete small and fake detected spots
  pix = 10 # pixel margin to the ZOI bounding box
  try:
    image = plt.imread(Liste_image[0])[:,:,0]
    im_mask = plt.imread(Mask)[:,:,0]/255.
  except IndexError:
    image = plt.imread(Liste_image[0])
    im_mask = plt.imread(Mask)[:,:,0]/255.
  ################################ First step : spotting the nodes
  img = invert(difference_of_gaussians(image, 5, 6))
  thresh = threshold_otsu(img[np.where(im_mask == 1)])
  imgb = img>thresh
  imgb[np.where(im_mask ==0)] = 0
  #plt.figure(); plt.imshow(imgb);plt.show()
  ################################ Labellization and ZOI detection
  label_img=label(255.*imgb) ### Labellization on binarized invariant minus 2 pixel of borders
  regions = regionprops((label_img))  ### identification of region props in all ROI
  boundbox=np.zeros_like(regions)
  barx=np.zeros_like(regions)
  bary=np.zeros_like(regions)
  bar = np.zeros_like(regions)
  areas=np.zeros_like(regions)
  for i, region in enumerate(regions): # save number of label, initial bbox and barycenter coordinates in vectors
    boundbox[i]=region.bbox
    barx[i], bary[i]=region.centroid
    areas[i]=region.area
    bar[i] = region.centroid
  vire = np.where((areas <100)|(areas>1800)) # je vire les points trop petits et les trop gros
  boundbox=np.delete(boundbox,vire)
  bar=np.delete(bar,vire)
  areas=np.delete(areas,vire)
  barx = np.delete(barx, vire)
  bary = np.delete(bary, vire)
  ##### Visual checking
  fig, ax = plt.subplots()
  ax.imshow(image, cmap = 'gray')
  for i in range(0,len(boundbox)): # for each ZOI
    gx,gy = bar[i] # barycenter coordinate of each ZOI
    minr, minc, maxr, maxc=boundbox[i] # bbox of each ZOI
    ax.plot(gy,gx, 'ro', markersize=6)
    minr, minc, maxr, maxc = boundbox[i]
    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=2)
    ax.add_patch(rect)
  all_px=np.vstack([barx]) # je cree des tableaux dans lequel je stocke les positions en X et en Y
  all_py=np.vstack([bary])
  plt.show()
  ##### Second step - iteration : ROI localization on the following images
  for j in np.arange(1, len(Liste_image), 1):  
    print(j)
    image = plt.imread(Liste_image[j])
    img = invert(difference_of_gaussians(image, 5,6))
#    img = invert(image)
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')    
    for i in range(len(areas)):
      '''
      Using previous image information, we loop over every previous regions.
      Previous bbox are increased and binarized to detect the main spot.  
      '''
      if math.isnan(boundbox[i][0]):
        '''
        if a spot has disappeared (out of the frame...)
        barycenter and bbox are set to 'nan' . They are excluded from 
        computation at this step.
        New values are set to 'nan' as well
        '''
        barx[i] = np.float('nan')
        bary[i] = np.float('nan')
        boundbox[i] = (np.float('nan'), np.float('nan'), np.float('nan'),np.float('nan'))
      else :
        minr, minc, maxr, maxc = boundbox[i] # bbox of each ZOI
        if minr < pix:
          minr = pix
        if minc < pix:
          minc = pix
        invar_ZOI = (img[minr-pix:maxr+pix, minc-pix:maxc+pix])
        thresh = threshold_otsu(invar_ZOI)
        invar_ZOI = invar_ZOI>thresh
        label_img = label(255.*invar_ZOI)
        regions = regionprops((label_img))
        area = [region.area for region in regions]
        if area:
          '''
          if spots are detected in the ZOI, the bigger one is selected.
          Barycenters and bounding box of this spot are then updated
          '''
          roi_index = np.where(area==max(area))[0][0]
          px,py = regions[roi_index].centroid # X,Y coordin. in local ZOI
          ppx = minr-pix+px # compute X coordinate in global image
          ppy = minc-pix+py # compute Y coordinate in global image
          minrr, mincc, maxrr, maxcc = regions[roi_index].bbox
          boundbox[i] = (minrr+minr-pix, mincc+minc-pix, maxrr+minr-pix, maxcc+minc-pix) # update bbox
          barx[i] = ppx # update X barycenter coordinate
          bary[i] = ppy # update Y barycenter coordinate
        else:
          ''' 
          if no spot is detected, boundbox and barycenter coordinates 
          are set to 'nan'
          '''
          boundbox[i]=(np.float('nan'),np.float('nan'), np.float('nan'),np.float( 'nan'))
          barx[i]=np.float('nan')
          bary[i]=np.float('nan')
        ax.plot(ppy,ppx,'ro',markersize=1)
#    plt.savefig(savefile + '/img_%06d.png'%j,dpi=150)
    plt.close()
    all_px = np.vstack([all_px, barx]) 
    ## add updated X coord of all ZOI to previous ones
    all_py = np.vstack([all_py, bary]) 
    ## add updated Y coord of all updated ZOI to previous ones
  #np.savetxt('../opti_angle_calib/2023_07_06/40d_cd/px_right.txt', all_px) # save X
  #np.savetxt('../opti_angle_calib/2023_07_06/40d_cd/py_right.txt', all_py) # save Y
  return all_px, all_py 	

def f(all_pxl, all_pyl, all_pxr, all_pyr):
  LA_allp = []
  for i in range(len(all_pxl)):
    A_allp = np.zeros((2,len(all_pxl[0]),2))
    for j in range(len(all_pxl[0])):
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


    spform = 332
    date = "2025_05_05"
    saving_folder = f'../{date}/results_calib/Spform_{spform}/'

    A111 = np.load(saving_folder+'A111.npy')
    A_pol = np.load(saving_folder+'A_pol.npy')
    
    M = np.load(f'../{date}/results_calib/transfomatrix.npy')

    #all_pxl, all_pyl = CoordCam(f'../data/SC37_40_4DFIXNR/left_SC37_40_4DFIXNR/', 'maskL.tiff', './test_calib/calib/images_centres_L')
    #np.savetxt(saving_folder + 'all_pxl.txt', all_pxl)
    #np.savetxt(saving_folder + 'all_pyl.txt', all_pyl)

    #all_pxr, all_pyr = CoordCam(f'../data/SC37_40_4DFIXNR/right_SC37_40_4DFIXNR/', 'maskR.tiff', './test_calib/calib/images_centres_R')
    #np.savetxt(saving_folder + 'all_pxr.txt', all_pxr)
    #np.savetxt(saving_folder + 'all_pyr.txt', all_pyr)

    all_pxl = np.loadtxt(saving_folder+'all_pxl.txt', delimiter=' ')
    all_pyl = np.loadtxt(saving_folder+'all_pyl.txt', delimiter=' ')
    all_pxr = np.loadtxt(saving_folder+'all_pxr.txt', delimiter=' ')
    all_pyr = np.loadtxt(saving_folder+'all_pyr.txt', delimiter=' ')
    Lp = f(all_pxl, all_pyl, all_pxr, all_pyr)
   
#LA BONNE IDEE
    Lrp = RtoL_transfo(Lp[0][1], M)
    Lfalse = []
    for j in range(len(Lrp)):
#      if Lp[0][0][j][0] - Lrp[j][0] > 40 or Lp[0][0][j][0] - Lrp[j][0] < 20: #pour 9 degrés et 10 degrés l=9cm
#      if Lp[0][0][j][0] - Lrp[j][0] > 90 or Lp[0][0][j][0] - Lrp[j][0] < 70: #pour 18 degrés
#      if Lp[0][0][j][0] - Lrp[j][0] > 60 or Lp[0][0][j][0] - Lrp[j][0] < 40: #pour 20 degrés l=15 cm
#      if Lp[0][0][j][0] - Lrp[j][0] > 115 or Lp[0][0][j][0] - Lrp[j][0] < 90: #pour 28 degrés
#      if Lp[0][0][j][0] - Lrp[j][0] > 80 or Lp[0][0][j][0] - Lrp[j][0] < 60: #pour 30 degrés l=20 cm
#      if Lp[0][0][j][0] - Lrp[j][0] > 140 or Lp[0][0][j][0] - Lrp[j][0] < 120: #pour 40 degrés
      if Lp[0][0][j][0] - Lrp[j][0] > 165 or Lp[0][0][j][0] - Lrp[j][0] < 130: #pour 40 degrés l=31 cm
#      if Lp[0][0][j][0] - Lrp[j][0] > 190 or Lp[0][0][j][0] - Lrp[j][0] < 140:
        Lfalse.append([Lrp[j], j])
    print(len(Lfalse))
    Lid = []
    for j in range(len(Lfalse)):
        for k in range(len(Lfalse)):
#          if Lp[0][0][Lfalse[j][1]][0] - Lfalse[k][0][0] > 10 and Lp[0][0][Lfalse[j][1]][0] - Lfalse[k][0][0] < 50: #pour 9 degrés et 10 degrés l=9cm
#          if Lp[0][0][Lfalse[j][1]][0] - Lfalse[k][0][0] > 60 and Lp[0][0][Lfalse[j][1]][0] - Lfalse[k][0][0] < 100: #pour 18 degrés
#          if Lp[0][0][Lfalse[j][1]][0] - Lfalse[k][0][0] > 40 and Lp[0][0][Lfalse[j][1]][0] - Lfalse[k][0][0] < 60: #pour 20 degrés l=15 cm
#          if Lp[0][0][Lfalse[j][1]][0] - Lfalse[k][0][0] > 80 and Lp[0][0][Lfalse[j][1]][0] - Lfalse[k][0][0] < 120: #pour 28 degrés
#          if Lp[0][0][Lfalse[j][1]][0] - Lfalse[k][0][0] > 60 and Lp[0][0][Lfalse[j][1]][0] - Lfalse[k][0][0] < 90: #pour 30 degrés l=20 cm
#          if Lp[0][0][Lfalse[j][1]][0] - Lfalse[k][0][0] > 110 and Lp[0][0][Lfalse[j][1]][0] - Lfalse[k][0][0] < 150: #pour 40 degrés
          if Lp[0][0][Lfalse[j][1]][0] - Lfalse[k][0][0] > 120 and Lp[0][0][Lfalse[j][1]][0] - Lfalse[k][0][0] < 180: #pour 40 degrés l=31 cm
#            if abs(Lp[0][0][Lfalse[j][1]][1] - Lfalse[k][0][1]) < 2: #pour 9 degrés et 20 degrés l=15 cm et 10 degrés l=9cm
#            if abs(Lp[0][0][Lfalse[j][1]][1] - Lfalse[k][0][1]) < 3: #pour 18 degrés
#            if abs(Lp[0][0][Lfalse[j][1]][1] - Lfalse[k][0][1]) < 4: #pour 28 degrés et 30 degrés l=20 cm
            if abs(Lp[0][0][Lfalse[j][1]][1] - Lfalse[k][0][1]) < 8: #pour 40 degrés et pour 40 degrés l=31 cm
              Lid.append([Lfalse[j][1], Lfalse[k][1]])
    print(len(Lid))
    for i in range(len(Lp)):
      Rightbuff = Lp[i][1].copy()
      for j in range(len(Lid)):
        Lp[i][1][Lid[j][0]] = Rightbuff[Lid[j][1]]


    
    Lx3d = []
    Ly3d = []
    Lz3d = []
    for i in range(len(Lp)):
      Left, Right = Lp[i]
      xSoloff_solution = pcs.Soloff_identification (Left,
                                                    Right,
                                                    A111,
                                                    A_pol,
                                                    Soloff_pform = spform,
                                                    method = 'curve_fit')
      x,y,z = xSoloff_solution
      Lx3d.append(x)
      Ly3d.append(y)
      Lz3d.append(z)
 
    np.savetxt(f'../{date}/SC37_40_4DFIXNR/Spform_{spform}/X3d_SC37_40.txt', Lx3d)
    np.savetxt(f'../{date}/SC37_40_4DFIXNR/Spform_{spform}/Y3d_SC37_40.txt', Ly3d)
    np.savetxt(f'../{date}/SC37_40_4DFIXNR/Spform_{spform}/Z3d_SC37_40.txt', Lz3d)

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
