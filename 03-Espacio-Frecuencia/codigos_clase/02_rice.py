import cv2

# Leer en escala de grisess
img = cv2.imread(
    "/Users/pyanezs/Documents/procesamiento-imagenes/Fotos/rice.png",
    cv2.IMREAD_GRAYSCALE)

cv2.imshow("Imagen Original", img)
cv2.waitKey(0)

# Difuminar
J = cv2.blur(img, (5, 5))

# Binarizacion
ret, BJ = cv2.threshold(J, 120, 255, type=cv2.THRESH_BINARY)

cv2.imshow("Imagen binarizada", BJ)
cv2.waitKey(0)

