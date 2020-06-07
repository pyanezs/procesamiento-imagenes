import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import cv2
import numpy as np

from skimage.filters import threshold_otsu
from skimage import measure


img = cv2.imread('rice.png')
N = 25
# MORPH_CROSS, MORPH_ELLIPSE, MORPH_RECT
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(N,N))

#apertura
tmp = cv2.erode(img,kernel)
apertura = cv2.dilate(tmp,kernel)

resta = cv2.subtract(img,apertura)

gray = resta[:,:,0]

# Find otsu threshold		
thresh = threshold_otsu(gray)
binary = gray > thresh

"""
fig, axes = plt.subplots(ncols=2, figsize=(12, 6.5))
ax = axes.ravel()
ax[0] = plt.subplot(1, 2, 1)
ax[1] = plt.subplot(1, 2, 2, sharex=ax[0], sharey=ax[0])

ax[0].imshow(gray, cmap="gray")
ax[0].set_title('Original')
ax[1].imshow(binary, cmap= "gray")
ax[1].set_title('Binaria')

plt.show()
"""

# componentes conectados
all_labels = measure.label(binary)

plt.figure()
plt.imshow(all_labels)

# Threshold data
centroids = []
areas = []

fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(binary, cmap="gray")

for region in measure.regionprops(label_image=all_labels):

    cx, cy = region.centroid[0], region.centroid[1]
    idx= region.image
    
    areas.append(region.area)
    centroids.append((cx, cy))
    if region.area>200:
        
        plt.scatter(cy, cx, marker="x", color="red", s=20)
        minr, minc, maxr, maxc = region.bbox
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
     

ax.set_axis_off()
plt.tight_layout()


plt.figure()
counts, bins = np.histogram(areas, bins=20)
plt.hist(bins[:-1], bins, weights=counts)

plt.show()
