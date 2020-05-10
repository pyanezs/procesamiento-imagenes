import cv2

img = cv2.imread(
    "/Users/pyanezs/Documents/procesamiento-imagenes/Fotos/cameraman.png")

img_median = cv2.medianBlur(img, 13)
cv2.imshow("Imagen", img_median)
cv2.waitKey(0)
