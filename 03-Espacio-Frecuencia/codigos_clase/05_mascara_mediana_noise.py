import cv2
import numpy as np

img = cv2.imread(
    "/Users/pyanezs/Documents/procesamiento-imagenes/Fotos/cameraman.png",
    cv2.IMREAD_GRAYSCALE)

m, n = img.shape

mat_noise = np.random.rand(m, n)


sp_noise_white = np.uint8(np.where(mat_noise >= 0.8, 255 ,0))
sp_noise_black = np.uint8(np.where(mat_noise >= 0.1, 1 ,0))

noise_img = cv2.add(img, sp_noise_white)
noise_img = cv2.multiply(noise_img, sp_noise_black)


img_median = cv2.medianBlur(noise_img, 3)


cv2.imshow("Imagen Ruido", noise_img)
cv2.imshow("Imagen", img_median)

cv2.waitKey(0)
