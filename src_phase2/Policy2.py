import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

'''
Policy network for two actions
'''

class Policy1(nn.Module):
    def __init__(self):
        super(Policy1, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        #64x64x3 => 32x32x32
            
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        #32x32x32 => 16x16x64
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        #16x16x64 => 8x8x64
        
        self.drop_out = nn.Dropout()
        # for brightness
        self.head1_fc1 = nn.Linear(8 * 8 * 64, 512)
        #self.BatchNorm2 = nn.BatchNorm1d(512)
        self.head1_fc2 = nn.Linear(512,40) # output actions can be one out of 40
        
        # for contrast
        self.head2_fc1 = nn.Linear(8 * 8 * 64, 512)
        self.head2_fc2 = nn.Linear(512,40)
        

        self.saved_log_probs1 = []
        self.saved_log_probs2 = []
        self.rewards = []
        
        
    def forward(self, x):
        #x = F.relu(self.affine1(x))
        #action_scores = self.affine2(x)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        #out = self.BatchNorm1(out)
        head1 = self.head1_fc1(out)
        action_scores1 = self.head1_fc2(head1)
        head2 = self.head2_fc1(out)
        action_scores2 = self.head2_fc2(head2)
        
        
        #normalise the scores
        action_scores1/=action_scores1.sum()
        action_scores2/=action_scores2.sum()
        #print(action_scores1)
        return [F.softmax(action_scores1, dim=1),F.softmax(action_scores2, dim=1)]
            
            