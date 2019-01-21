import argparse
from ssd_pytorch.ssdDetector import Detector
import cv2

parser = argparse.ArgumentParser(description='PyTorch REINFORCE')
parser.add_argument('--show', type=int, default=0, metavar='G',
                    help='to show detections in the image')
parser.add_argument('--ID', type=int, default=1234, metavar='N',
                    help='Image ID in VOC Dataset')

                    
args = parser.parse_args()

image=cv2.imread('ssd_pytorch/data/example.jpg')
# d = Detector(img_id=args.ID,show=args.show)
d = Detector(show=0)
f,g = d.detect(image=image)
print('Name',f)
print('k',g)