import cv2
import glob
import sys
import numpy as np
#load images
filename = 'fields.jpg'
img=cv2.imread(filename)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gray=cv2.resize(img_gray,(256,256))



#resize image
scaling_factor = 0.5
img_scaled = cv2.resize(img_gray, None, fx=scaling_factor, fy=scaling_factor,interpolation=cv2.INTER_LINEAR)
#h, w = img.shape


'''#histogram equalization for higher contrast
img_scaled = cv2.equalizeHist(img_scaled)
img_hist=img_scaleds'''

#edge detection
sobel_horizontal = cv2.Sobel(img_scaled, cv2.CV_64F, 1, 0, ksize=5)
sobel_vertical = cv2.Sobel(img_scaled, cv2.CV_64F, 0, 1, ksize=5)
laplacian = cv2.Laplacian(img_scaled, cv2.CV_64F)
canny = cv2.Canny(img_scaled, 50, 240)

#display image
# while(1):
cv2.imshow('orignal',img)
cv2.imshow('grayscale',img_scaled)
cv2.imshow('edge detected',canny)
cv2.waitKey()