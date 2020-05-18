import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


img = cv2.imread('cameraman.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray_norm = cv2.normalize(gray.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

noise = np.random.random(gray.shape)*0.1
output = gray_norm + noise

cv2.imshow('ruido gaussiano', output)
cv2.waitKey(0)
