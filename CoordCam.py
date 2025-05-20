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
    plt.show()
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
        ax.imshow(img, cmap='gray')
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
                ax.plot(ppy,ppx,'ro',markersize=1)
        #if j==136:
        #plt.show()
        plt.savefig(savefile + '/img_%06d.png'%j,dpi=150)
        plt.close()
        all_px = np.vstack([all_px, barx])
        ## add updated X coord of all ZOI to previous ones
        all_py = np.vstack([all_py, bary])
        ## add updated Y coord of all updated ZOI to previous ones
        #  np.savetxt('./2023_08_29/40d_cd/SC37_40_P7NR/px_right.txt', all_px) # save X
        #  np.savetxt('./2023_08_29/40d_cd/SC37_40_P7NR/py_right.txt', all_py) # save Y
    return all_px, all_py


date = '2025_05_15'
sample = 'SC37_40_A1L'
saving_folder=f'./{date}/{sample}/'



###reverse the right images, cameras are in mirror
#Liste_image  = sorted(glob(f'./{date}/video_extenso_right/'+"0*"))
#for image in Liste_image:
    #img=cv2.imread(image)
    #img=cv2.rotate(img,cv2.ROTATE_180)
    #cv2.imwrite(image,img)

all_pxl, all_pyl = CoordCam(f'./{date}/video_extenso_left/', 'maskL.tiff',saving_folder)
np.save(saving_folder + 'all_pxl.npy', all_pxl)
np.save(saving_folder + 'all_pyl.npy', all_pyl)

all_pxr, all_pyr = CoordCam(f'./{date}/video_extenso_right/', 'maskR.tiff',saving_folder)
np.save(saving_folder + 'all_pxr.npy', all_pxr)
np.save(saving_folder + 'all_pyr.npy', all_pyr)
