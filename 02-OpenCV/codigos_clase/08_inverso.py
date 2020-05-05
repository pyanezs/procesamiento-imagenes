import cv2

img = cv2.imread("/Users/pyanezs/Documents/procesamiento-imagenes/Fotos/cameraman.png")

# Image has three channels , with same values = grey-scale image
print(img[5, 5, :])

# Revert colors
neg = 255 - img

# Display image
cv2.imshow("Image", neg)
cv2.waitKey(0)

