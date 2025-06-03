import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from skimage.filters import threshold_otsu, difference_of_gaussians
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border
from skimage.util import invert
from glob import glob
import os
import sys
import pathlib
import cv2
from math import *
import math



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
    if os.path.exists(savefile) :
        ()
    else :
        P = pathlib.Path(savefile)
        pathlib.Path.mkdir(P, parents = True)

    Liste_image  = sorted(glob(path+"0*")) #list of image
    #  Liste_image = Liste_image[::7]
    Mask = path + mask # mask defined from the first image
    First_img = 1
    Last_img = 2000
    impair_coeff = 1. # impairs the thresh level for global binarization of spots
    Min_area = 10 #used to delete small and fake detected spots
    pix = 10 # pixel margin to the ZOI bounding box
    image_raw = plt.imread(Liste_image[0])
    if image_raw.ndim == 3:
        image = image_raw[:,:,0]  # ou moyenne des canaux: np.mean(image_raw, axis=2)
        ndimm = 3
    else:
        image = image_raw
        ndimm = 2
    mask_raw = plt.imread(Mask)
    if mask_raw.ndim == 3:
        im_mask = mask_raw[:,:,0]/255.
    else:
        im_mask = mask_raw/255.
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
    vire = np.where((areas <100)|(areas>1000)) # je vire les points trop petits et les trop gros
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
    #plt.show()
    ##### Second step - iteration : ROI localization on the following images
    for j in np.arange(1, len(Liste_image), 1):
        print(j)
        if ndimm==3:
            image = plt.imread(Liste_image[j])[:,:,0]
        else:
            image = plt.imread(Liste_image[j])
        img = invert(difference_of_gaussians(image, 5,6))
        #img = invert(image)
        fig, ax = plt.subplots()
        #ax.imshow(img, cmap='gray')
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
                #if invar_ZOI.ndim == 3:
                    #invar_ZOI = rgb2gray(invar_ZOI)
                thresh = threshold_otsu(invar_ZOI)
                invar_ZOI = invar_ZOI>thresh
                label_img = label(255.*invar_ZOI)
                regions = regionprops((label_img))
                area = [region.area for region in regions]
                if area:
                    '''difference_of_gaussians
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
                ax.axis('off')
                ax.plot(ppy,ppx,'ko', markersize=5)
        #if j==136:
        #plt.show()
        #plt.savefig(savefile + 'img_%06d.png'%j,dpi=150)
        plt.close()
        all_px = np.vstack([all_px, barx])
        ## add updated X coord of all ZOI to previous ones
        all_py = np.vstack([all_py, bary])
        ## add updated Y coord of all updated ZOI to previous ones
        #  np.savetxt('./2023_08_29/40d_cd/SC37_40_P7NR/px_right.txt', all_px) # save X
        #  np.savetxt('./2023_08_29/40d_cd/SC37_40_P7NR/py_right.txt', all_py) # save Y
    return all_px, all_py

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


date = '2025_06_02'
sample = 'SC37_20'
saving_folder=f'./{date}/{sample}/'



###reverse the right images, cameras are in mirror
#Liste_image  = sorted(glob(f'./{date}/{sample}/video_extenso_right/'+"0*"))
#for image in Liste_image:
    #img=cv2.imread(image)
    #img=cv2.rotate(img,cv2.ROTATE_180)
    #cv2.imwrite(image,img)


all_pxl, all_pyl = CoordCam(saving_folder+'video_extenso_left/', 'maskL.tiff',saving_folder+'ROI_left/')
np.save(saving_folder + 'all_pxl.npy', all_pxl)
np.save(saving_folder + 'all_pyl.npy', all_pyl)

all_pxr, all_pyr = CoordCam(saving_folder+'video_extenso_right/', 'maskR.tiff',saving_folder+'ROI_right/')
np.save(saving_folder + 'all_pxr.npy', all_pxr)
np.save(saving_folder + 'all_pyr.npy', all_pyr)



Lp = f(all_pxl, all_pyl, all_pxr, all_pyr)
N = len(Lp[0][0])

### Appariement des points
Template = cv2.imread(saving_folder + 'Template_L.tiff') #Template Gimp
Y_template,X_template = 943,946 #Constantes à modifier en fonction des coordonnées du Template

Img = cv2.imread(saving_folder + 'Template_R.tiff') #Image 0 de la caméra de droite

#On fait correspondre le template sur l'image de droite et on observe le décalage entre les deux images
result = cv2.matchTemplate(Img, Template, cv2.TM_CCOEFF)
(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)
(startX,startY) = maxLoc
CoordTemplate = (Y_template, X_template, Y_template + Template.shape[0], X_template + Template.shape[1])
Diff = (CoordTemplate[0] - maxLoc[0],CoordTemplate[1] - maxLoc[1])


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
col_ind = np.zeros(N,dtype = int)
for i in range(N): # On fait l'appariement en regardant les points les plus proches après compensation du delta
    m = np.linalg.norm(Matched_Left[i]-Matched_Right[0])
    Id = 0
    for j in range(N):
        if np.linalg.norm(Matched_Left[i] - Matched_Right[j]) < m :
            #if np.all(np.not_equal(col_ind,j)):
            m = np.linalg.norm(Matched_Left[i] - Matched_Right[j])
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
    print(matched_gauche[i][1] - matched_droite[i][1])

# Appariement
for i in range(len(Lp)):
    Rightbuff = Lp[i][1].copy()
    for j in range(len(col_ind)):
        Lp[i][1][row_ind[j]] = Rightbuff[col_ind[j]]

np.save(saving_folder + 'Lp.npy',Lp)
