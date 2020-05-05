import cv2

img = cv2.imread("/Users/pyanezs/Documents/procesamiento-imagenes/Fotos/lena.png")

# Info foto
dimensions = img.shape

height = img.shape[0]
width = img.shape[1]
# Single channel pictures do not have a third element in the shape variable
# This could cause an excution error
channels = img.shape[2]

print(f"Dimensions: {dimensions}")
print(f"Height: {height}")
print(f"Width: {width}")
print(f"channels: {channels}")

# Select a single pixel
# img[x,y,channel]
pixel = img[265,327,:]
print(f"Pixel 265,327")
print(pixel)

# Select a section of the photo
# roi: region of interest
roi = img[255:281, 314:348, :]
cv2.imshow("ROI", roi)
cv2.waitKey()

# Select a section of the photo with specific channel
# roi: region of interest
roi = img[255:281, 314:348, 2]
cv2.imshow("ROI-Ch Red", roi)
cv2.waitKey()





