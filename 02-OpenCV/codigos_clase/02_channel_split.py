
import cv2

img = cv2.imread("/Users/pyanezs/Documents/procesamiento-imagenes/Fotos/lena.png")

# 1. param: Nombre de ventana
# 2. param: variable con image
cv2.imshow("image", img)

# Wait for key stroke to close window
cv2.waitKey(0)

# Split into channels: Open CV by default refers
# to the channels as BGR (Blue, Green, Red) instead
# of RGB (Red, Green, Blue)
ch_blue, ch_green, ch_red = cv2.split(img)

# opencv is able to open multiple windows at the time
cv2.imshow("CH Blue", ch_blue)
cv2.imshow("CH Green", ch_green)
cv2.imshow("CH Red", ch_red)

cv2.waitKey(0)
