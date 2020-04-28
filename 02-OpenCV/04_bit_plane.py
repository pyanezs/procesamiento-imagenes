import cv2 
import numpy as np

img = cv2.imread('cameraman.png')
img = img[:,:,0]  #ocupamos en canal 0
o = []
for k in range(0, 8):
    plane = np.full((img.shape[0], img.shape[1]), 2 ** k, np.uint8)
	
    res = plane & img	
    x = res*255
    o.append(x)
    cv2.imshow("plane", x)
    cv2.waitKey()

cv2.imshow('all',np.hstack(o))
cv2.waitKey()
cv2.destroyAllWindows()


