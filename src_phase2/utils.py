import numpy as np
import pandas as pd
import random
from PIL import Image,ImageEnhance
import cv2
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
    new_width = size
    new_height = size
    img = Image.open(filepath)
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
    
def letterbox_image(img, inp_dim):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_CUBIC)
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image
    
    return canvas,new_w,new_h
    
def synthetic_change(img,action_table_synth):
    action_bright_synth = random.choice(action_table_synth) #choose a random change in the image(synthetic)
    change_img = change_brightness(img,action_bright_synth)

    #change the color
    # action_color_synth = random.choice(action_table_synth)
    # change_img = change_color(change_img,action_color_synth)
    #
    return change_img,action_bright_synth
    
    
def get_iou(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
 
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
 
 
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
 
	# return the intersection over union value
	return iou
    
def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names
    
def getResizedBB(arr,w,h,width,height,reso):
    if height>width:
        xmin=arr[0]*h/height
        ymin=arr[1]*w/width+(reso-w)/2
        xmax=arr[2]*h/height
        ymax=arr[3]*w/width+(reso-w)/2
        
    elif height<=width:
        xmin=arr[0]*w/width+(reso-h)/2
        ymin=arr[1]*h/height
        # print(h/height)
        xmax=arr[2]*w/width+(reso-h)/2
        ymax=arr[3]*h/height
        
    return [xmin,ymin,xmax,ymax]
    
    
def get_F1(truth,pred,iou_threshold):
    '''
    Returns True Positives, False Positives,
    False Negatives, IOU_array of detected images
    truth:[xmin,ymin,xmax,ymax,width,height,class]
    pred:[xmin,ymin,xmax,ymax,class]
    '''
    TP = 0
    FP = 0
    FN = 0
    dft = pd.DataFrame(truth)
    iou_arr = []
    names = dft[6].unique() #has all the unique class(labels) names
    # iterate through all classes/labels
    for i in range(len(names)):
        label_df = dft[dft[6]==names[i]]
        for bbox_orig in np.array(label_df):
            iou_temp = []
            for bbox_pred in pred:
                if len(pred)>0:
                    iou_temp.append(get_iou(bbox_orig[0:4],bbox_pred[0:4].astype(float)))
            if len(pred)>0: # this is because of error when length of pred becomes zero
                overlap = np.max(iou_temp)
                index = np.argmax(iou_temp)
            else:
                overlap=0
            if overlap>iou_threshold:
                if pred[int(index)][-1]==names[i]: # check whether same class or not
                    iou_arr.append(overlap)
                    pred = np.delete(pred,(index),axis=0) #pop out the matched boxes
                    
                    TP+=1
                else:
                    FN+=1
            else:
                FP+=1
    return TP,FP,FN,iou_arr