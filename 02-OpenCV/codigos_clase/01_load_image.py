import cv2

img = cv2.imread("/Users/pyanezs/Documents/procesamiento-imagenes/Fotos/lena.png")

# 1. param: Nombre de ventana
# 2. param: variable con image
cv2.imshow("image", img)

# Wait for key stroke to close window
cv2.waitKey(0)
