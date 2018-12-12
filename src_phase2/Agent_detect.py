'''
RL Agent to change image brightness according
to image(state) to get maximum performance from
a pre-trained network, in this case YOLO
Agent network is ResNet18 with REINFORCE Algorithm
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
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--load_model',type=int,default=0,metavar='N',
                    help='whether to use the saved model or not, to resume training')
parser.add_argument('--weights',default='Weights_det.pt',
                    help='Path to the weights.')
parser.add_argument('--optimizer',default='Optimizer_det.pt',
                    help='Path to the Optimizer.')
parser.add_argument('--threshold',default=0.85,type=float,
                    help='Threshold for the Similarity Index')
parser.add_argument('--show',default=0,
                    help='To show the images before and after transformation')
parser.add_argument('--episodes',type=int,default=10,
                    help='Number of episodes')
parser.add_argument('--lr',type=float,default=1e-6,
                    help='Learning rate')
parser.add_argument('--std',default=0,
                    help='To print standard deviation of the probabilities')
parser.add_argument('--confidence',type=float,default=0.5,
                    help='Confidence Threshold for Object Detection(YOLO)')
parser.add_argument('--nms_thresh',type=float,default=0.3,
                    help='Non Maximal Suppression Threshold for YOLO')
parser.add_argument('--reso',type=int,default=416,
                    help='Resolution of Image to be fed into YOLO for detection keeping the Aspect Ration constant')
parser.add_argument('--image_filepath',type=str,default='VOCdevkit/VOC2007/',
                    help='VOC Dataset')
parser.add_argument('--iou_threshold',type=float,default=0.5,
                    help='Threshold for IOU to determine if object is detected or not')
parser.add_argument('--alpha',type=float,default=0.5,
                    help='IOU Weight for reward --> r=alpha*(iou)+(1-alpha)*F1')
                    
                    
args = parser.parse_args()

df=pd.read_csv('labels.csv')
image_filepath=args.image_filepath+str('JPEGImages/') #train images are here
annotations_filepath=args.image_filepath+str('Annotations/') # test images are here
num_images=len(os.listdir(image_filepath))  # total number of images in the dataset
action_table_synth=np.linspace(0.05,2,40) # to synthetically change the images
action_table=1/action_table_synth # the optimal actions are reciprocal of the factor of the synthesized image
# 0 means complete dark,2 means complete bright, 1 means the original image

###########################################################################

policy = resnet18()  # can also be resnet34, resnet50, resnet101, resnet152
if args.load_model==0:
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)
eps = np.finfo(np.float32).eps.item()
classes = load_classes('data/coco.names')

CUDA = torch.cuda.is_available()
if CUDA:
    print('CUDA available, setting GPU mode')
    policy.cuda()


print('Loading the model if any')
print(args.load_model)
if args.load_model==1:
    policy.load_state_dict(torch.load(args.weights))
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)
    optimizer.load_state_dict(torch.load(args.optimizer))
    print(args.weights)
    print(args.optimizer)
    
def save_model():
    print('Saving the weights')
    torch.save(policy.state_dict(),args.weights)
    torch.save(optimizer.state_dict(),args.optimizer)

def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs1 = policy(state)
    #print(probs1)
    m1 = Categorical(probs1)
    if args.std:
        print(probs1.std())
    action1 = m1.sample()
    policy.saved_log_probs1.append(m1.log_prob(action1))
    #print(m1.log_prob(action1))
    return action1.item()

def finish_episode():
    # print('Finishing Episode')
    R = 0
    policy_loss1 = []
    rewards = []
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)
    rewards = torch.tensor(rewards)
    #rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    for log_prob, reward in zip(policy.saved_log_probs1, rewards):
        policy_loss1.append(-log_prob * reward)
        #print(policy_loss1)
    
    optimizer.zero_grad()
    policy_loss1 = torch.cat(policy_loss1).sum()
    #print(policy_loss1)
    #print(policy_loss2)
    policy_loss = policy_loss1
    #print(policy_loss)
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs1[:]
    
image_list = os.listdir(image_filepath)
image_list = shuffle_arr(image_list) #shuffle_arr from utils.py shuffles the array randomly
exit=1
def main():
    # for episodes in range(args.episodes):
    for episodes in range(num_images):
        img_name = image_list[episodes]
        # index_img = np.random.uniform(low = 0, high = num_images, size = (1,)).astype(int) # random number between 0 and num_images eg: 435
        # img_name = os.listdir(image_filepath)[index_img[0]] # eg: 000005.jpg
        img = Image.open(image_filepath+img_name)
        orig_img_arr = np.array(img)
        orig_img_arr_lr, w, h = letterbox_image(orig_img_arr,(args.reso,args.reso)) #resized image by keeping aspect ratio same
        x = Image.fromarray(orig_img_arr_lr.astype('uint8'), 'RGB')
        orig_img_arr=np.array(x)

        ## mean size of image is (384,472,3)
        # img = read(folder+str(index_img[0])+'.png',64)
        #synthesize the image
        change_img , act = synthetic_change(img,action_table_synth)
        #convert to array
        change_img_arr = np.array(change_img)
        change_img_arr, w, h= letterbox_image(change_img_arr,(args.reso,args.reso))
        x = Image.fromarray(change_img_arr.astype('uint8'), 'RGB')
        change_img_arr=np.array(x)
        #img_arr = np.array(img)
        state=change_img_arr/255
        state = (np.reshape(state,(3,state.shape[0],state.shape[1]))) # to get into pytorch mode
        #take action acording to the state
        bright_action=select_action(state)
        agent_act=action_table[bright_action]
        #synthetically changed image is rectified by the agent
        new_image = change_brightness(change_img,action_table[bright_action])
        # new_image = change_color(new_image,action_table[color_action])

        # feed the changed image to detector
        new_image_arr=np.array(new_image)
        new_image_arr, w, h = letterbox_image(new_image_arr,(args.reso,args.reso))
        x = Image.fromarray(new_image_arr.astype('uint8'), 'RGB')
        new_image_arr=np.array(x)
        if args.show:
            plt.imshow(orig_img_arr)
            plt.title('Original')
            plt.show()
            plt.imshow(change_img_arr)
            plt.title('Modified image')
            plt.show()
            plt.imshow(new_image_arr)
            plt.title('Agent modified image')
            plt.show()
        # print('Getting detections')
        detector = Detector(args.confidence,args.nms_thresh,args.reso,new_image_arr)
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


        TP,FP,FN,iou = get_F1(resized_gnd_truth_arr,pred_arr,args.iou_threshold)
        # reward=np.mean(IOU+F1_score) #to make sure everything is in 0-1 range
        recall = TP/(TP+FN+eps)
        precision = TP/(TP+FP+eps)
        F1 = 2*recall*precision/(precision+recall+eps)
        if len(iou)>0: #### if no detections then iou=[]
            iou_reward = np.mean(iou)
        else:
            iou_reward = 0
        reward = args.alpha*(iou_reward)+(1-args.alpha)*F1
        print(f'Episode:{episodes}\t Reward:{reward}')
        policy.rewards.append(reward)
        finish_episode()  # does all backprop
        print_arg=False
        if print_arg:
            print(f'F1:{F1}')
            print(f'Reward:{reward}')
            print(f'Action:{act}')
            print(f'Agent Action:{agent_act}')
            print(f'Ideal action:{1/act}')
            print()

    save_model()
    
    
if __name__ == '__main__':
    main()