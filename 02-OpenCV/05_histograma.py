import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('cameraman.png')

counts, bins = np.histogram(img)
plt.hist(bins[:-1], bins, weights=counts)

plt.show()
