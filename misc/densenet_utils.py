import torch
import torch.nn as nn
import torch.nn.functional as F

class myDensenet(nn.Module):
    def __init__(self, densenet):
        super(myDensenet, self).__init__()
        self.densenet = densenet

    def forward(self, img, att_size=14):
        x = img.unsqueeze(0)
        #print x.size()
        x = self.densenet.features(x)
        #print("Att size")
        #print att.size()
        #print("Pooling")
        att = F.adaptive_avg_pool2d(x,[att_size,att_size]).squeeze().permute(1, 2, 0)
        #att = att.squeeze(0)
        #att=att.permute(1,2,0)
        return att

