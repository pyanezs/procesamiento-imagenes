import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread(
    '/Users/pyanezs/Documents/procesamiento-imagenes/Fotos/cameraman.png')

counts, bins = np.histogram(img, bins = 100)
plt.hist(bins[:-1], bins, weights=counts)

plt.show()
