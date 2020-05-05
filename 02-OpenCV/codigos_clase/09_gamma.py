import cv2
import numpy as np


def gamma_correction(img, factor):
    """Simple function that applies gamma function to a image"""
    img = img / 255.0 # Scale image
    img = cv2.pow(img, factor) # Applies pow function to each pixel
    return np.uint8(img * 255) # De-scale and return and numpy.uint8

# Read image
img = cv2.imread("/Users/pyanezs/Documents/procesamiento-imagenes/Fotos/cameraman.png")

# Apply gamma correction function
output = gamma_correction(img, 0.1)

# Display image
cv2.imshow("Image", output)
cv2.waitKey(0)
