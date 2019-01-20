import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from data import VOC_CLASSES as labels
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
import cv2
from matplotlib import pyplot as plt
from data import VOCDetection, VOC_ROOT, VOCAnnotationTransform
from ssd import build_ssd

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

net = build_ssd('test', 300, 21)    # initialize SSD
net.load_weights('weights/ssd300_mAP_77.43_v2.pth')


# here we specify year (07 or 12) and dataset ('test', 'val', 'train')
testset = VOCDetection(VOC_ROOT, [('2007', 'val')], None, VOCAnnotationTransform())
img_id = 345  # change here for image id
image, image_name = testset.pull_image(img_id)
print(image_name)
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# View the sampled input image before transform
# plt.figure(figsize=(6,6))
# plt.imshow(rgb_image)
# plt.show()


def base_transform(image):
    x = cv2.resize(image, (300, 300)).astype(np.float32)
    x -= (104.0, 117.0, 123.0)
    x = x.astype(np.float32)
    x = x[:, :, ::-1].copy()
    # plt.imshow(x)
    x = torch.from_numpy(x).permute(2, 0, 1)
    return x
x = base_transform(image)

xx = Variable(x.unsqueeze(0))     # wrap tensor in Variable
if torch.cuda.is_available():
    xx = xx.cuda()
y = net(xx)

plt.figure(figsize=(6,6))
colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
plt.imshow(rgb_image)  # plot the image for matplotlib
currentAxis = plt.gca()

detections = y.data
# scale each detection back up to the image
scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)
arr=[]
for i in range(detections.size(1)):
    j = 0
    while detections[0,i,j,0] >= 0.6:
        # arr.append(detections[0,i,:,0])
        score = detections[0,i,j,0]
        label_name = labels[i-1]
        display_txt = '%s: %.2f'%(label_name, score)
        pt = (detections[0,i,j,1:]*scale).cpu().numpy()
        
        coords = (pt[0], pt[1]), pt[2]-pt[0], pt[3]-pt[1]
        color = colors[i]
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor':color, 'alpha':0.5})
        j+=1
        print(pt)
        print(label_name)
        
plt.show()

