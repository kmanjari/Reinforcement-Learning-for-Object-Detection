from __future__ import division
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from util import *
from darknet import Darknet
from preprocess import inp_to_image
import pandas as pd
import random
import argparse
import pickle as pkl
import matplotlib.pyplot as plt

def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network.
    
    Returns a Variable
    """

    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img, w, h = (letterbox_image(orig_im, (inp_dim, inp_dim)))
    img_ = img[:,:,::-1]
    img_= img_.transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim, w, h

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




def write(x, img):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    color = random.choice(colors)
    cv2.rectangle(img, c1, c2,color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
    return img

def arg_parse():
    """
    Parse arguements to the detect module
    
    """
    
    
    parser = argparse.ArgumentParser(description='YOLO v3 Cam Demo')
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.3)
    parser.add_argument("--reso", dest = 'reso', help =
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "160", type = str)
    return parser.parse_args()

class Detector():
    def __init__(self,confidence,nms_thresh,reso,image):
        self.cfgfile = "cfg/yolov3.cfg"
        self.weightsfile = "yolov3.weights"
        self.num_classes = 80
        self.confidence = confidence
        self.nms_thresh = nms_thresh
        self.reso = reso
        self.image = image
    def detector(self):
        CUDA = torch.cuda.is_available()
        bbox_attrs = 5 + self.num_classes
        model = Darknet(self.cfgfile)
        model.load_weights(self.weightsfile)
        model.net_info["height"] = self.reso
        inp_dim = int(model.net_info["height"])
        assert inp_dim % 32 ==0
        assert inp_dim >32
        if CUDA:
            model.cuda()
        model.eval()
        img, orig_im, dim ,f,g= prep_image(self.image,inp_dim)
        im_dim = torch.FloatTensor(dim).repeat(1,2)
        
        if CUDA:
            im_dim = im_dim.cuda()
            img = img.cuda()
        # print(img)
    
        output = model(Variable(img), CUDA)
        # print('Output')
        # print((output))
        output = write_results(output, self.confidence, self.num_classes, nms = True, nms_conf = self.nms_thresh)
        if type(output) == int:
            output = torch.tensor(np.zeros((1,8)))
        
        output[:,1:5] = torch.clamp(output[:,1:5], 0.0, float(inp_dim))/float(inp_dim)
     
#            im_dim = im_dim.repeat(output.size(0), 1)
        output[:,[1,3]] *= self.image.shape[1]
        output[:,[2,4]] *= self.image.shape[0]

    
        classes = load_classes('data/coco.names')
        colors = pkl.load(open("pallete", "rb"))
    
        # list(map(lambda x: write(x, orig_im), output))
        
        # print(output)
        return output
        # im=np.reshape(img.numpy()[0,:,:,:],(inp_dim,inp_dim,3))
        
        
# d=Detector(0.5,0.3,416,image)
# d.detector()

    
    

