import torch

from ssdDetector import Detector

d = Detector(123)
f = d.detect()

print(f)
