'''
RL Agent to change image brightness according
to image(state) to get maximum performance from
a pre-trained network, in this case SSD
Agent network is ResNet50 with REINFORCE Algorithm
Reward = alpha*(iou)+(1-alpha)*F1
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
from resnet_policy import resnet34
from resnet_policy import resnet50

# gets home dir cross platform
HOME = os.path.expanduser("~")
print('home:',HOME)

from ssd_pytorch.ssdDetector import Detector
import pandas as pd
import os.path


parser = argparse.ArgumentParser(description='PyTorch REINFORCE SSD')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=1234, metavar='N',
                    help='random seed (default: 123)')
parser.add_argument('--load_model',type=int,default=0,metavar='N',
                    help='whether to use the saved model or not, to resume training')
parser.add_argument('--weights',default='Weights_ssd_50_newReward.pt',
                    help='Path to the weights.')
parser.add_argument('--optimizer',default='Optimizer_ssd_50_newReward.pt',
                    help='Path to the Optimizer.')
parser.add_argument('--show',default=0,
                    help='To show the images before and after transformation')
parser.add_argument('--episodes',type=int,default=10,
                    help='Number of episodes')
parser.add_argument('--lr',type=float,default=1e-6,
                    help='Learning rate')
parser.add_argument('--std',default=0,
                    help='To print standard deviation of the probabilities')
parser.add_argument('--image_filepath',type=str,default='VOCdevkit/VOC2007/',
                    help='VOC Dataset')
parser.add_argument('--iou_threshold',type=float,default=0.5,
                    help='Threshold for IOU to determine if object is detected or not')
parser.add_argument('--epoch',type=int,default=1,
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
#################
# changed on 19/02/2019
action_table_synth=np.linspace(0.3,1.7,29)
action_table=np.linspace(0.3,1.7,15)
#################
# 0 means complete dark,2 means complete bright, 1 means the original image

###########################################################################

# policy = resnet18()  # can also be resnet34, resnet50, resnet101, resnet152
# policy = resnet34()
print('Weights Name: ',args.weights)
print('Optimizer Name: ',args.optimizer)
policy = resnet50()
if args.load_model==0:
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)
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
    if CUDA:
        state = state.cuda()
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
    if CUDA:
        rewards = rewards.cuda()
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
    
# Detector init
detector = Detector(show=args.show) # initialize the SSD

print()
print('Running for %d epochs'%args.epoch)
print()

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

            ## mean size of image is (384,472,3)
            #synthesize the image
            change_img , act = synthetic_change(img,action_table_synth,1)
            #convert to array
            change_img_arr = np.array(change_img)
            state=change_img_arr
            state = (np.reshape(state,(3,state.shape[0],state.shape[1]))) # to get into pytorch mode
            #take action acording to the state
            bright_action=select_action(state)
            agent_act=action_table[bright_action]
            #synthetically changed image is rectified by the agent
            new_image = change_brightness(change_img,action_table[bright_action])
            # new_image = change_color(new_image,action_table[color_action])

            # feed the changed image to detector
            new_image_arr=np.array(new_image)
            
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
                
            img_fed = cv2.cvtColor(new_image_arr, cv2.COLOR_RGB2BGR)
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
            policy.rewards.append(reward)
            finish_episode()  # does all backprop
            print_arg=False
            if print_arg:
                print('F1:%f'%(F1))
                print('Reward:%f'%(reward))
                print('Action:%f'%(act))
                print('Agent Action:%f'%agent_act)
                print('Ideal action:%f'%(1/act))
                print()

        save_model()
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
