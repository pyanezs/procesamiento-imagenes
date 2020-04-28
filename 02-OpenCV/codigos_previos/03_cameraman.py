import cv2
import numpy as np

def gamma_correction(img, factor):
    img = img/255.0
    img = cv2.pow(img, factor)
    return np.uint8(img*255)
    
img = cv2.imread('cameraman.png')
output = gamma_correction(img, 0.4)
cv2.imshow('gamma',output)
cv2.waitKey(0)
cv2.destroyAllWindows()