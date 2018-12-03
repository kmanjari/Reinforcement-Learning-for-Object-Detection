'''
Code for 4 actions training
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
from Policy1 import *
from utils import *
from skimage.measure import compare_ssim as ssim
from tqdm import tqdm


parser = argparse.ArgumentParser(description='PyTorch REINFORCE')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=1234, metavar='N',
                    help='random seed (default: 123)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--load_model',type=int,default=1,metavar='N',
                    help='whether to use the saved model or not to resume training')
parser.add_argument('--weights',default='Weights.pt',
                    help='Path to the weights.')
parser.add_argument('--optimizer',default='Optimizer.pt',
                    help='Path to the Optimizer.')
parser.add_argument('--threshold',default=0.85,
                    help='Threshold for the Similarity Index')
parser.add_argument('--show',default=0,
                    help='To show the images before and after transformation')
parser.add_argument('--episodes',type=int,default=10,
                    help='Number of episodes')
parser.add_argument('--tmax',type=int,default=100,
                    help='Number of images per episode')
parser.add_argument('--lr',type=float,default=1e-6,
                    help='Learning rate')
parser.add_argument('--std',default=0,
                    help='To print standard deviation of the probabilities')
                    
args = parser.parse_args()


#env = gym.make('CartPole-v0')
#env.seed(args.seed)

####26th October commented random seed line below
torch.manual_seed(args.seed)

###########################################################################

folder='train/'
image_size =64
img = Image.open('train/6.png')
new_width  = image_size
new_height = image_size
img = img.resize((new_width, new_height), Image.ANTIALIAS)
img=change_brightness(img,0.5)
#plt.imshow(img)

a=list(img.size)
a.append(3)
a=tuple(a)


x=img##### FILL with image as input
action_table_synth=np.linspace(0.05,2,40) # to synthetically change the images
action_table=1/action_table_synth # the optimal actions are reciprocal of the factor of the synthesized image
# 0 means complete dark,2 means complete bright, 1 means the original image

n_actions = len(action_table) # increase or decrease brightness
state_dim = a #### fill image size with channels

###########################################################################

policy = Policy1()

if args.load_model==0:
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)
eps = np.finfo(np.float32).eps.item()


print('Loading the model if any')
print(args.load_model)
if args.load_model==1:
    policy.load_state_dict(torch.load(args.weights))
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)
    optimizer.load_state_dict(torch.load(args.optimizer))
    print(args.weights)
    print(args.optimizer)
    #pass


def save_model():
    #print('Saving the weights')
    #policy.save_state_dict('Weights1.pt')
    torch.save(policy.state_dict(),args.weights)
    torch.save(optimizer.state_dict(),args.optimizer)


def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    [probs1,probs2,probs3,probs4] = policy(state)
    #print(probs1)
    m1 = Categorical(probs1)
    m2 = Categorical(probs2)
    m3 = Categorical(probs3)
    m4 = Categorical(probs4)
    if args.std:
        print(probs1.std())
        print(probs2.std())
        print(probs3.std())
        print(probs4.std())
    action1 = m1.sample()
    action2 = m2.sample()
    action3 = m3.sample()
    action4 = m4.sample()
    policy.saved_log_probs1.append(m1.log_prob(action1))
    policy.saved_log_probs2.append(m2.log_prob(action2))
    policy.saved_log_probs3.append(m3.log_prob(action3))
    policy.saved_log_probs4.append(m4.log_prob(action4))
    #print(m1.log_prob(action1))
    return [action1.item(),action2.item(),action3.item(),action4.item()]


def finish_episode():
    R = 0
    policy_loss1 = []
    policy_loss2 = []
    policy_loss3 = []
    policy_loss4 = []
    rewards = []
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)
    rewards = torch.tensor(rewards)
    #rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    for log_prob, reward in zip(policy.saved_log_probs1, rewards):
        policy_loss1.append(-log_prob * reward)
        #print(policy_loss1)
    for log_prob, reward in zip(policy.saved_log_probs2, rewards):
        policy_loss2.append(-log_prob * reward)
        #print(policy_loss2)
    for log_prob, reward in zip(policy.saved_log_probs3, rewards):
        policy_loss3.append(-log_prob * reward)
    for log_prob, reward in zip(policy.saved_log_probs4, rewards):
        policy_loss4.append(-log_prob * reward)
    
    optimizer.zero_grad()
    policy_loss1 = torch.cat(policy_loss1).sum()
    policy_loss2 = torch.cat(policy_loss2).sum()
    policy_loss3 = torch.cat(policy_loss3).sum()
    policy_loss4 = torch.cat(policy_loss4).sum()
    #print(policy_loss1)
    #print(policy_loss2)
    #print(policy_loss3)
    #print(policy_loss4)
    policy_loss = policy_loss1 + policy_loss2 + policy_loss3 + policy_loss4
    #print(policy_loss)
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs1[:]
    del policy.saved_log_probs2[:]
    del policy.saved_log_probs3[:]
    del policy.saved_log_probs4[:]


def main():
    for episodes in (range(args.episodes)):
        #one_reward=0  # to count number of +1 rewards in a set of training
        #print(episodes)
        t_max = args.tmax
        one_reward=0
        for t in tqdm(range(t_max)):  # Don't put an infinite loop while learning
            #print('Session Number:',t)
            #select the image to be artificially changed
            index_img = np.random.uniform(low = 1, high = 10000, size = (t_max,)).astype(int)
            img = read(folder+str(index_img[t])+'.png',64)
            #print(folder+str(index_img[t])) #print the image filepath
            
            #synthesize the image
            # change the brightness
            action_bright_synth = random.choice(action_table_synth) #choose a random change in the image(synthetic)
            change_img = change_brightness(img,action_bright_synth)
            #change the contrast
            action_contrast_synth = random.choice(action_table_synth)
            change_img = change_contrast(change_img,action_contrast_synth)
            #change the sharpness
            action_sharpness_synth = random.choice(action_table_synth)
            change_img = change_sharpness(change_img,action_contrast_synth)
            #change the color
            action_color_synth = random.choice(action_table_synth)
            change_img = change_color(change_img,action_color_synth)
            
            #convert to array
            change_img_arr=np.array(change_img)
            #img_arr = np.array(img)
            state = change_img_arr/255
            
            
            [bright_action,contrast_action,sharpness_action,color_action] = select_action(np.reshape(state,(3,64,64)))
            #print('Action',bright_action,contrast_action,sharpness_action,color_action)
            #print('\n')
            #print('Action:',action_table[action],action_synth)
            
            #take action acording to the state
            new_image = change_brightness(img,action_table[bright_action])
            new_image = change_contrast(new_image,action_table[contrast_action])
            new_image = change_sharpness(new_image,action_table[sharpness_action])
            new_image = change_color(new_image,action_table[color_action])
            
            if args.show:
                plt.imshow(np.array(change_img_arr))
                plt.show()
                plt.imshow(np.array(new_image))
                plt.show()
                
            #compare the images and get similarity values
            m1,m2,m3,s=compare_images(np.array(img),np.array(new_image))
            #print('Mse: %f %f %f  and SI: %f'%(m1,m2,m3,s))
            #give rewards
            if s>=args.threshold:
                reward=s-(m1+m2+m3)/10000
                one_reward+=1
                #print('Reward: ',reward)
            else:
                reward=-(m1+m2+m3)/10000
                #print('NegReward: ',reward)
            '''
            print(folder+str(index_img[t])) #print the image filepath
            print('\n')
            print('Action:',action_table[action],action_synth)
            print('Similarity is ',s)
            print('Reward:',reward)
            '''
            #reward = int(input("Enter Reward: "))
            
            # to calculate total number of +1 rewards per set of training(100 images)
            policy.rewards.append(reward)
            finish_episode()
            
        save_model()
        print('%d number of +1 rewards '%(one_reward))
        print('*'*50)
            
    
if __name__ == '__main__':
    main()