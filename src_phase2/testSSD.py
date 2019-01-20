import argparse
from ssd_pytorch.ssdDetector import Detector

parser = argparse.ArgumentParser(description='PyTorch REINFORCE')
parser.add_argument('--show', type=int, default=0, metavar='G',
                    help='to show detections in the image')
parser.add_argument('--ID', type=int, default=1234, metavar='N',
                    help='Image ID in VOC Dataset')

                    
args = parser.parse_args()


d = Detector(img_id=args.ID,show=args.show)
d = Detector(img_id=0,show=1)
f,g = d.detect()
print('Name',f)
print('k',g)