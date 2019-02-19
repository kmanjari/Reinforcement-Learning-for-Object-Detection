'''
Get score of SSD with random changes in the image
'''
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image,ImageEnhance
import os
import torch
import torch.optim as optim
from torch.distributions import Categorical
from utils import *
from tqdm import tqdm
from resnet_policy import resnet18
from ssd_pytorch.ssdDetector import Detector
import pandas as pd

parser = argparse.ArgumentParser(description='PyTorch REINFORCE SSD')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=1234, metavar='N',
                    help='random seed (default: 123)')
parser.add_argument('--show',default=0,
                    help='To show the images before and after transformation')
parser.add_argument('--episodes',type=int,default=10,
                    help='Number of episodes')
parser.add_argument('--std',default=0,
                    help='To print standard deviation of the probabilities')
parser.add_argument('--image_filepath',type=str,default='VOCdevkit/VOC2007/',
                    help='VOC Dataset')
parser.add_argument('--iou_threshold',type=float,default=0.5,
                    help='Threshold for IOU to determine if object is detected or not')
parser.add_argument('--epoch',type=int,default=5,
                    help='Number of epochs')
parser.add_argument('--alpha',type=float,default=0.1,
                    help='weight for IOU in reward')
args = parser.parse_args()

df=pd.read_csv('labels.csv')
image_filepath=args.image_filepath+str('JPEGImages/') #train images are here
annotations_filepath=args.image_filepath+str('Annotations/') # test images are here
num_images=len(os.listdir(image_filepath))  # total number of images in the dataset
# action_table_synth=np.linspace(0.05,2,40) # to synthetically change the images
# action_table=1/action_table_synth # the optimal actions are reciprocal of the factor of the synthesized image
# 0 means complete dark,2 means complete bright, 1 means the original image
#################
# changed on 19/02/2019
action_table_synth=np.linspace(0.3,1.7,29)
action_table=np.linspace(0.3,1.7,15)
#################
###########################################################################

policy = resnet18()  # can also be resnet34, resnet50, resnet101, resnet152
eps = np.finfo(np.float32).eps.item()
classes = load_classes('data/coco.names')

CUDA = torch.cuda.is_available()
if CUDA:
   print('CUDA available, setting GPU mode')
   print('GPU Name:',torch.cuda.get_device_name(0))
   print('Memory Usage:')
   # print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
   print('Allocated:', torch.cuda.memory_allocated(0)/1024**3, 'GB')
   print('Cached:   ', torch.cuda.memory_cached(0)/1024**3, 'GB')
   # print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')
   policy.cuda()


    
# Detector init
detector = Detector(show=args.show) # initialize the SSD

def main():
    image_list = os.listdir(image_filepath)
    #num_images=10
    reward_epoch=[]
    for epoch in range(args.epoch):
        image_list = shuffle_arr(image_list) #shuffle_arr from utils.py shuffles the array randomly
        reward_arr=[]
        for episodes in tqdm(range(num_images)):
        # for episodes in tqdm(range(3)):
            img_name = image_list[episodes]
            img = Image.open(image_filepath+img_name)
            orig_img_arr = np.array(img)

            change_img , act = synthetic_change(img,action_table_synth,1)
            #convert to array
            change_img_arr = np.array(change_img)
            
            if args.show:
                plt.imshow(orig_img_arr)
                plt.title('Original')
                plt.show()
                plt.imshow(change_img_arr)
                plt.title('Modified image')
                plt.show()
                
            img_fed = cv2.cvtColor(change_img_arr, cv2.COLOR_RGB2BGR)
            pred_box = detector.detect(image=img_fed)
            pred_arr=[]
            for i in range(len(pred_box[0])):
                box=list(pred_box[0][i])
                box.append(pred_box[1][i])
                pred_arr.append(box)
            # get ground truths
            ground_truth_df = df[df['filename'] == img_name]
            ground_truth_df = ground_truth_df.drop('filename', axis = 1)
            ground_truth_arr = np.array(ground_truth_df)
            
            # get F1 Score
            # get IOU average of all detected objects
            pred_arr = np.array(pred_arr)
            TP,FP,FN,iou = get_F1(ground_truth_arr,pred_arr,args.iou_threshold)
            # reward=np.mean(IOU+F1_score) #to make sure everything is in 0-1 range
            iou=np.array(iou)
            recall = TP/(TP+FN+eps)
            precision = TP/(TP+FP+eps)
            F1 = 2*recall*precision/(precision+recall+eps)
            if len(iou)>0:
                # reward=np.sum(iou)/(TP+FP+FN)
                reward = args.alpha*(np.mean(iou)) + (1-args.alpha)*(F1)
            else:
                reward=0
            reward_arr.append(reward)
            
            print_arg=False
            if print_arg:
                print('F1:%f'%(F1))
                print('Reward:%f'%(reward))
                print('Action:%f'%(act))
                print('Agent Action:%f'%agent_act)
                print('Ideal action:%f'%(1/act))
                print()

        print('#'*50)
        print()
        print('Epochs:%d'%epoch)
        print('Mean reward:%f'%(np.mean(reward_arr)))
        print()
        print('#'*50)
        reward_epoch.append(np.mean(reward_arr))
    print('Reward array:',reward_epoch)
    
if __name__ == '__main__':
    main()
