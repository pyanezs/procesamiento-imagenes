import cv2
v = 12
img = cv2.imread('cameraman.png', cv2.IMREAD_GRAYSCALE)
imgN = cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
imgA = cv2.blur(imgN,(v,v))
imgB = cv2.blur(imgN*imgN,(v,v))
out = imgB - imgA**2

out = cv2.normalize(out.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
cv2.imshow('image',out)
cv2.waitKey(0)
