import cv2

img = cv2.imread(
    "/Users/pyanezs/Documents/procesamiento-imagenes/Fotos/cameraman.png")

mask = (20,20)

# Normalizar imagen
img_norm = cv2.normalize(img.astype("float"), None, 0.0, 1.0, cv2.NORM_MINMAX)

# Mascara a imagen
img_a = cv2.blur(img_norm, mask)

# Mascara a imagen ^ 2
img_b = cv2.blur(img_norm * img_norm, mask)

cv2.imshow("Imagen A", img_a)
cv2.imshow("Imagen B", img_b)

# Imagen de salida
img_c = img_b - img_a ** 2 # E[X ^ 2] - (E[X]) ^ 2
img_c = cv2.normalize(img_c.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

cv2.imshow("out", img_c)

cv2.waitKey(0)
