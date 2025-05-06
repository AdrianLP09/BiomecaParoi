import cv2
import matplotlib.pyplot as plt

date='2025_03_24'

img = cv2.imread(f'./{date}/r/000001_4.036.tiff')
#rows,cols = img.shape[:2]

# cols-1 and rows-1 are the coordinate limits.
img=cv2.rotate(img,cv2.ROTATE_180)
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
#cv2.imshow("Image",img)
#cv2.waitKey(0)
cv2.imwrite(f'./{date}/r/000001_4.036transfo.tiff',img)
