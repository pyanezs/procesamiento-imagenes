#LEAME:
#archivo lena.png debe encontrarse en la misma carpeta de este script
import cv2 

img = cv2.imread('coke_can.jpg')
cv2.imshow('mi ventana', img)
cv2.waitKey(0)