import cv2
import glob
import sys
import numpy as np
import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import data, exposure


#load images
filename = 'Shepherds_Purse'
for var_img in glob.glob(filename+'/*.*'):
    try :
        img=cv2.imread(var_img,cv2.IMREAD_UNCHANGED)

        #resize image
        '''scaling_factor = 0.3
        img_scaled = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor,interpolation=cv2.INTER_LINEAR)
        h, w = img.shape'''
        img_grayscale=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
       
        img_scaled=cv2.resize(img_grayscale,(256,256))


        #histogram equalization for higher contrast
        img_scaled = cv2.equalizeHist(img_scaled)
        

        #edge detection
        sobel_horizontal = cv2.Sobel(img_scaled, cv2.CV_64F, 1, 0, ksize=5)
        sobel_vertical = cv2.Sobel(img_scaled, cv2.CV_64F, 0, 1, ksize=5)
        laplacian = cv2.Laplacian(img_scaled, cv2.CV_64F)
        canny = cv2.Canny(img_scaled, 50, 240)

        #Histogram of Oriented Gradients
        fd, hog_image = hog(img_scaled, orientations=8, pixels_per_cell=(16, 16),cells_per_block=(1, 1), visualize=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

        ax1.axis('off')
        ax1.imshow(img_scaled, cmap=plt.cm.gray)
        ax1.set_title('Input image')

        # Rescale histogram for better display
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

        ax2.axis('off')
        ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
        ax2.set_title('Histogram of Oriented Gradients')
        plt.show()

        #display image
       # while(1):
        cv2.imshow('orignal',img_scaled)
        cv2.imshow('edge detected',canny)
        cv2.waitKey()
    except Exception as e:
            print (e)