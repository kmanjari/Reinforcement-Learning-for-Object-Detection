import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from .data import VOC_CLASSES as labels
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
import cv2
from matplotlib import pyplot as plt
from .data import VOCDetection, VOC_ROOT, VOCAnnotationTransform
from .ssd import build_ssd

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')



class Detector():
    def __init__(self,weight_path='ssd_pytorch/weights/ssd300_mAP_77.43_v2.pth',show=0):
        self.weight_path = weight_path  # weight path for SSD
        self.show = show  # Boolean to show images
        self.net = build_ssd('test', 300, 21)    # initialize SSD
        self.net.load_weights(self.weight_path)
        # self.image = image
        
    def base_transform(self,image):
        x = cv2.resize(image, (300, 300)).astype(np.float32)
        x -= (104.0, 117.0, 123.0)
        x = x.astype(np.float32)
        x = x[:, :, ::-1].copy()
        # plt.imshow(x)
        x = torch.from_numpy(x).permute(2, 0, 1)
        return x

    def detect(self,image):
        # here we specify year (07 or 12) and dataset ('test', 'val', 'train')
        # testset = VOCDetection(VOC_ROOT, [('2007', 'val')], None, VOCAnnotationTransform())
        # image,image_name = testset.pull_image(self.img_id)
        # print(image_name)
        # rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        x = self.base_transform(image)

        xx = Variable(x.unsqueeze(0))    # wrap tensor in Variable
        if torch.cuda.is_available():
            xx = xx.cuda()
        y = self.net(xx)
        
        if self.show:
            plt.figure(figsize=(6,6))
            colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
            plt.imshow(rgb_image)  # plot the image for matplotlib
            currentAxis = plt.gca()
            
        detections = y.data
        # scale each detection back up to the image
        scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)
        arr=[]
        
        boxes=[]
        detected_label=[]
        for i in range(detections.size(1)):
            j = 0
            while detections[0,i,j,0] >= 0.6:
                # arr.append(detections[0,i,:,0])
                score = detections[0,i,j,0]
                label_name = labels[i-1]
                display_txt = '%s: %.2f'%(label_name, score)
                pt = (detections[0,i,j,1:]*scale).cpu().numpy()
                coords = (pt[0], pt[1]), pt[2]-pt[0], pt[3]-pt[1]
                
                if self.show:
                    color = colors[i]
                    currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
                    currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor':color, 'alpha':0.5})
                j+=1
                # print(pt)
                boxes.append(pt)
                # print(label_name)
                detected_label.append(label_name)
        if self.show:
            plt.show()
        
        return boxes,detected_label
