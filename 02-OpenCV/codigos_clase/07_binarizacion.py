import cv2

img = cv2.imread("/Users/pyanezs/Documents/procesamiento-imagenes/Fotos/lena.png")

# Select area and specific channel
roi = img[255:281, 314:348, 2]

# Make a subselection
sub_roi = roi[1:20, 15:25]

rt, bina = cv2.threshold(sub_roi, 150, 255, cv2.THRESH_BINARY)

cv2.imshow("Image", sub_roi)
cv2.waitKey(0)

cv2.imshow("Image", bina)
cv2.waitKey(0)

