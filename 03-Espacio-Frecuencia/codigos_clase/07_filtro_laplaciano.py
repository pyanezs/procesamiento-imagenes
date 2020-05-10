import cv2


img = cv2.imread(
    "/Users/pyanezs/Documents/procesamiento-imagenes/Fotos/rice.png")

lpl = cv2.Laplacian(img, cv2.CV_8U)

u_sharp = cv2.subtract(img, lpl)

cv2.imshow("img", img)
cv2.imshow("sharp", u_sharp)

cv2.waitKey(0)
