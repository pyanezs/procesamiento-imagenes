import cv2

img = cv2.imread(
    "/Users/pyanezs/Documents/procesamiento-imagenes/Fotos/lena.png")

src = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

dst = cv2.equalizeHist(src)

cv2.imshow("Src Image", src)
cv2.waitKey(0)

cv2.imshow("Eq Image", dst)
cv2.waitKey(0)
