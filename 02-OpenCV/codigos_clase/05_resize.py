import cv2

img = cv2.imread("/Users/pyanezs/Documents/procesamiento-imagenes/Fotos/lena.png")
roi = img[255:281, 314:348, :]

factor_times = 10
height = int(roi.shape[0] * factor_times)
width = int(roi.shape[1] * factor_times)

# Please note that resize change the convetion of the shape
dim = (width, height)

# New resized matrix
roi_2 = cv2.resize(roi, dim, interpolation=cv2.INTER_AREA)

# Show image
cv2.imshow("image", roi_2)
cv2.waitKey()
