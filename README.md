## Project on using reinforcement learning agents to improve object detection performance
[Work done till now](https://drive.google.com/file/d/1RDYVBnBJZoxxKoaK5inetT7FC4paKiWF/view?usp=sharing)

Reward = sum(IOU)/(TP+FP+FN)
* SSD Baseline Reward(No change):0.687654
* SSD Random Changes Reward(Only brightness change):0.655650
* SSD Random Changes ALL(Brightness,contrast,sharpness and color change):0.586907

Reward First ten epochs:
Reward array: [0.6014174591995224, 0.6097525071709082, 0.6084995439643106, 0.6151870477374811, 0.6118711896559677, 0.6168747728792393, 0.6176739406658254, 0.6188732521751484, 0.6184163783265217, 0.6244941503894942]

Reward array: [0.6279089838918714, 0.62988880462949, 0.6235758075710192, 0.6251019727468605, 0.6303144931568306, 0.6264562762869332, 0.6292544845367566, 0.6264706063257051, 0.6337272964480785, 0.6325929772235952]

