import cv2

img = cv2.imread(
    "/Users/pyanezs/Documents/procesamiento-imagenes/Fotos/cameraman.png")

img_median = cv2.blur(img, (14, 14))

cv2.imshow("Imagen mediana", img_median)

cv2.waitKey(0)

