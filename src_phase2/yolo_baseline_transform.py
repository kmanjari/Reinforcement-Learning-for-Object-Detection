'''
To check average reward (weighted mean of iou and F1)
for YOLO without alteration in images.
'''
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image,ImageEnhance
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
#from Policy2 import *   ##### modified Policy2.py to Policy3.py
# from Policy2 import *
from utils import *
from skimage.measure import compare_ssim as ssim
from tqdm import tqdm
from resnet_policy import resnet18
from Yolo_detect import Detector
import pandas as pd

parser = argparse.ArgumentParser(description='PyTorch REINFORCE')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=1234, metavar='N',
                    help='random seed (default: 123)')
parser.add_argument('--confidence',type=float,default=0.5,
                    help='Confidence Threshold for Object Detection(YOLO)')
parser.add_argument('--nms_thresh',type=float,default=0.3,
                    help='Non Maximal Suppression Threshold for YOLO')
parser.add_argument('--reso',type=int,default=416,
                    help='Resolution of Image to be fed into YOLO for detection keeping the Aspect Ration constant')
parser.add_argument('--image_filepath',type=str,default='VOCdevkit/VOC2007/',
                    help='VOC Dataset')   # 'VOCdevkit_test/VOC2007/' for test
parser.add_argument('--iou_threshold',type=float,default=0.5,
                    help='Threshold for IOU to determine if object is detected or not')
parser.add_argument('--alpha',type=float,default=0.5,
                    help='IOU Weight for reward --> r=alpha*(iou)+(1-alpha)*F1')
parser.add_argument('--label_path',type=str,default='labels.csv',
                    help='Filepath to labels.csv')
parser.add_argument('--epochs',type=int,default=1,
                    help='Number of epochs')

args = parser.parse_args()
df=pd.read_csv('labels.csv')  # 'labels_test.csv' for test
image_filepath=args.image_filepath+str('JPEGImages/') #train images are here
annotations_filepath=args.image_filepath+str('Annotations/') # test images are here
num_images=len(os.listdir(image_filepath))  # total number of images in the dataset
classes = load_classes('data/coco.names')
eps = np.finfo(np.float32).eps.item()
action_table_synth=np.linspace(0.05,2,40) # to synthetically change the images

###########################################################################
    
def main():
    for epochs in range(args.epochs):
        reward_arr=[]
        iou_arr=[]
        F1_arr=[]
        TP_total=0
        FN_total=0
        FP_total=0
        gnd_total=0
        for episodes in tqdm(range(len(os.listdir(image_filepath)))):
        # for episodes in tqdm(range(2)):
            img_name = os.listdir(image_filepath)[episodes] # eg: 000005.jpg
            img = Image.open(image_filepath+img_name)
            orig_img_arr = np.array(img)
            orig_img_arr_lr, w, h = letterbox_image(orig_img_arr,(args.reso,args.reso)) #resized image by keeping aspect ratio same
            x = Image.fromarray(orig_img_arr_lr.astype('uint8'), 'RGB')
            orig_img_arr=np.array(x)
            
            ### change the image
            change_img , act = synthetic_change(img,action_table_synth,2)
            #convert to array
            change_img_arr = np.array(change_img)
            change_img_arr, w, h= letterbox_image(change_img_arr,(args.reso,args.reso))
            x = Image.fromarray(change_img_arr.astype('uint8'), 'RGB')
            change_img_arr=np.array(x)
            
            plt.imshow(orig_img_arr)
            plt.title('Original Image')
            plt.show()
            plt.imshow(change_img_arr)
            plt.title('Changed Image')
            plt.show()
            
            # plt.imshow(orig_img_arr)
            # plt.show()
            detector = Detector(args.confidence,args.nms_thresh,args.reso,change_img_arr)
            d=detector.detector()

            # get ground truths
            ground_truth_df = df[df['filename'] == img_name]
            ground_truth_arr=[]
            for i in range(len(ground_truth_df)):
                ground_truth_arr.append(ground_truth_df.iloc[i][1:])
            ground_truth_arr=np.array(ground_truth_arr)

            # rearrange gnd truth array to [xmin,ymin,xmax,ymax,width,height,class] and get resized bboxes
            resized_gnd_truth_arr=np.copy(ground_truth_arr)
            for i in range(len(ground_truth_arr)):
                arr=ground_truth_arr[i][:4]
                t=getResizedBB(arr,h,w,orig_img_arr.shape[0],orig_img_arr.shape[1],args.reso)
                resized_gnd_truth_arr[i][:4]=np.array(t)
                
            # rearrange predicted arrays and get class name from number
            pred = np.array(d)
            pred_arr=[]
            for i in range(len(pred)):
                arr=list(pred[i][1:5])
                arr.append(classes[int(pred[i][-1])])
                arr=np.array(arr)
                pred_arr.append(arr)
            pred_arr=np.array(pred_arr)

            # get F1 Score
            # get IOU average of all detected objects
            gd_len=len(resized_gnd_truth_arr)
            gnd_total+=gd_len
            TP,FP,FN,iou = get_F1(resized_gnd_truth_arr,pred_arr,args.iou_threshold)
            TP_total+=TP
            FN_total+=FN
            FP_total+=FP
            # reward=np.mean(IOU+F1_score) #to make sure everything is in 0-1 range
            recall = TP/(TP+FN+eps)
            precision = TP/(TP+FP+eps)
            F1 = 2*recall*precision/(precision+recall+eps)
            if len(iou)>0: #### if no detections then iou=[]
                iou_reward = np.mean(iou)
                iou_arr.append(np.mean(iou))
            else:
                iou_reward = 0
            reward = args.alpha*(iou_reward)+(1-args.alpha)*F1
            reward_arr.append(reward)
            
            # print(f'Episode:{episodes}\t Reward:{reward}')
            
        print('#'*50)
        print()
        print('Epoch:%d'%(epochs+1))
        print('Mean Reward:%f '%(np.mean(reward_arr)))
        print('Total True Positives:%d'%(TP_total))
        print('Total False Positives:%d'%(FP_total))
        print('Total False Negatives:%d'%(FN_total))
        print('Total ground truth images:%d'%(gnd_total))
        print('Mean IOU score:%f'%(np.mean(iou_arr)))
        print()
        print('#'*50)
        print()
        
    
if __name__ == '__main__':
    main()