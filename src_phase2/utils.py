import numpy as np
import random
from PIL import Image,ImageEnhance
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim as ssim

def change_brightness(image,brightness_factor):
    #change the brightness
    #brightness factor between (0,2) with 1 being original image
    brightness = ImageEnhance.Brightness(image)
    image = brightness.enhance(brightness_factor)
    return image
    
def change_contrast(image,contrast_factor):
    #change the contrast
    #contrast factor between (0,2) with 1 being original image
    contrast = ImageEnhance.Contrast(image)
    image = contrast.enhance(contrast_factor)
    return image

def change_color(image,color_factor):
    #change the color
    #color factor between (0,2) with 1 being original image
    color = ImageEnhance.Color(image)
    image = color.enhance(color_factor)
    return image

def change_sharpness(image,sharpness_factor):
    #change the sharpness
    #sharpness factor between (0,2) with 1 being original image
    sharpness = ImageEnhance.Sharpness(image)
    image = sharpness.enhance(sharpness_factor)
    return image
    
    
def read(filepath,size):
    '''
    Read the image with size (size x size x 3) from the filepath
    Example: img = read('/Image_for_training/',64) will read image
             with size (64 x 64 x 3)
    NOTE : Have to convert img to np.array if wanted in array form
    '''
    img = Image.open(filepath)
    new_width  = size
    new_height = size
    img = img.resize((new_width, new_height), Image.ANTIALIAS)
    return img
    
def mse(imageA, imageB):
    '''
	 the 'Mean Squared Error' between the two images is the
	 sum of the squared difference between the two images;
	 NOTE: the two images must have the same dimension
     '''
    err1 = np.sum((imageA.astype("float")[:,:,0] - imageB.astype("float")[:,:,0]) ** 2)
    err2 = np.sum((imageA.astype("float")[:,:,1] - imageB.astype("float")[:,:,1]) ** 2)
    err3 = np.sum((imageA.astype("float")[:,:,2] - imageB.astype("float")[:,:,2]) ** 2)
    err1 /= float(imageA.shape[0] * imageA.shape[1])
    err2 /= float(imageA.shape[0] * imageA.shape[1])
    err3 /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
    return err1,err2,err3



def compare_images(imageA, imageB,show=False):
    '''
     compute the mean squared error and structural similarity
     index for the images
    '''
    m1,m2,m3 = mse(imageA, imageB)
    s = ssim(imageA, imageB,multichannel=True)
    if show==True:
        fig=plt.figure()
        # setup the figure
        plt.suptitle("MSE: %.2f  %.2f  %.2f SSIM: %.2f" % (m1,m2,m3,s))

        # show first image
        ax = fig.add_subplot(1, 2, 1)
        plt.imshow(imageA, cmap = plt.cm.gray)
        plt.axis("off")

        # show the second image
        ax = fig.add_subplot(1, 2, 2)
        plt.imshow(imageB, cmap = plt.cm.gray)
        plt.axis("off")

        # show the images
        plt.show()

    return m1,m2,m3,s
    
    
def synthetic_change(img,action_table_synth):
    action_bright_synth = random.choice(action_table_synth) #choose a random change in the image(synthetic)
    change_img = change_brightness(img,action_bright_synth)

    #change the color
    action_color_synth = random.choice(action_table_synth)
    change_img = change_color(change_img,action_color_synth)
    
    return change_img