# this file removes green and red lines from iCBR images. It also reducestheir size.

import os
import cv2 as cv
import dataset_functions as cf # chromosome functions
import numpy as np 

path = os.getcwd()
filepath_original_images = os.path.join(path,'iCBR_Dataset\Original_Images') # path for the original metaphase images
filepath_mid_filtered_images = os.path.join(path, 'Synthetic_Dataset\iCBR_Original_Filtered\Original_Images_Without_Closing') # path for the mid filtered metaphase images
filepath_filtered_images = os.path.join(path,'Synthetic_Dataset\iCBR_Original_Filtered\Original_Images_With_Closing') # path for the filtered metaphase images
filenames = next(os.walk(filepath_original_images), (None, None, []))[2] # filenames of the original metaphase images

print(filenames)

for filename in filenames:
    print("\nFiltering Image {} \n".format(filename))
    filepath_image = filepath_original_images + "\\" + filename # filepath of the original image
    img_original = cv.imread(filepath_image) 
    # saves image mask
    img_mask = np.zeros((img_original.shape[0], img_original.shape[1]), dtype='uint8')

    
    # REMOVE GREEN AND RED COLORS (3rd channel is red, 2nd channel is green) 
    img_filtered = img_original.copy() # creates a new matrix, so it doesn't affect the original (safe environment)
    # iterates over the image pixels (i for rows, j for columns)
    for i in range(img_filtered.shape[0]):
        for j in range(img_filtered.shape[1]):
            # if all channels aren't equal, it makes that pixel white
            if img_filtered[i,j,0] != img_filtered[i,j,1] or img_filtered[i,j,0] != img_filtered[i,j,2] or img_filtered[i,j,1] != img_filtered[i,j,2]:
                img_filtered[i,j,0] = 255
                img_filtered[i,j,1] = 255
                img_filtered[i,j,2] = 255
                img_mask[i,j] = 255
    
    # BGR TO GRAYSCALE - folder "Original_Images_Without_Closing"
    img_filtered = img_filtered[:,:,0] # only saves one of the rgb channels, once the chromosomes are black and white
    filepath_mid_filtered_image = os.path.join(filepath_mid_filtered_images, filename)
    cv.imwrite(filepath_mid_filtered_image, img_filtered) # saves the filtered image
    
    # CLOSING OF GREEN LINES - Imagens_Filtradas_Closing
    # iterates over the image pixels (i for rows, j for columns)
    kernel = 2
    iterations = 3
    for it in range(iterations):
        for i in range(img_filtered.shape[0]):
            for j in range(img_filtered.shape[1]):
                if img_mask[i,j] == 255:
                    img_filtered[i,j] = cf.my_blur(img_filtered, [i,j], kernel)
    
    # First approach....             
    # for i in range(img_filtered.shape[0]):
    #     for j in range(img_filtered.shape[1]):
    #         # only checks pixels that are not the image's limits
    #         if (i!=0 and i!=img_filtered.shape[0]-1 and j!=0 and j!=img_filtered.shape[1]-1):
    #             # vertical closing - if the pixel is white and the above/under pixels are not white, it attributes that pixel a mean value of adjacent pixels
    #             if (img_filtered[i,j]==255 and img_filtered[i,j-1]!=255 and img_filtered[i,j+1]!=255):
    #                 img_filtered[i,j] = int((int(img_filtered[i,j-1]) + int(img_filtered[i,j+1]))/2)
    #             # horizontal closing - if the pixel is white and the lateral pixels are not white, it attributes that pixel a mean value of adjacent pixels
    #             elif (img_filtered[i,j]==255 and img_filtered[i-1,j]!=255 and img_filtered[i+1,j]!=255):
    #                 img_filtered[i,j] = int((int(img_filtered[i-1,j]) + int(img_filtered[i+1,j]))/2)
                    
    filepath_filtered_image = os.path.join(filepath_filtered_images, filename) # filepath for the filtered image
    cv.imwrite(filepath_filtered_image, img_filtered) # saves the filtered image
