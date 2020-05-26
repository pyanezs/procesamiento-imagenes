import cv2
import numpy as np

# img = cv2.imread(
#     '/Users/pyanezs/Documents/procesamiento-imagenes/Fotos/cameraman.png',
#     cv2.IMREAD_GRAYSCALE)
img = cv2.imread(
    '/Users/pyanezs/Documents/procesamiento-imagenes/Tarea_03/outs/p1/puente.jpg',
    cv2.IMREAD_GRAYSCALE)

m = img.shape[0]
delta = 15
V = np.fix(np.linspace(delta, m-delta, delta)).astype('uint8')
img[V, :] = img[V, :]+50
img[:, V] = img[:, V]+50
cv2.imshow('ruido', img)



cv2.waitKey()
