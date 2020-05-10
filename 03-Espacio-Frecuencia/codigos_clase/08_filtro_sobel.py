import cv2
from skimage import filters


img = cv2.imread(
    "/Users/pyanezs/Documents/procesamiento-imagenes/Fotos/cameraman.png")

# Transforma imagen a escala de grises
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sobel = filters.sobel(gray)

cv2.imshow("Sobel", sobel)

cv2.waitKey(0)
