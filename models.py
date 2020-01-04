from argparse import ArgumentParser
import torch
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
from torchvision.models import vgg16
import numpy as np
from scipy import stats
import sys
import os, yaml
sys.path.append('./utils/')
from utils.AttentionModule import *
from utils.resnet import resnet34

class BiNet(nn.Module):
    def __init__(self,weight_file = None,feature_channels = 512,n1_nodes = 1024,n2_nodes = 512):
        super(BiNet,self).__init__()
        resnet = resnet34(pretrained=False)
        if not(weight_file == None):
            resnet.load_state_dict(torch.load(weight_file),strict=False)
        self.features1 = nn.Sequential(resnet.conv1,resnet.bn1,resnet.relu,
        resnet.maxpool,resnet.layer1,resnet.layer2,
        resnet.layer3,resnet.layer4)
        self.conv2d = nn.Conv2d(3,3,3)
        vggnet = vgg16(pretrained = True)
        features = vggnet.features[:-1] 
        self.features2 = nn.Sequential(features[0:3],CBAM(64, 16),features[3:8],CBAM(128, 16),features[8:15],CBAM(256, 16),features[15:29],CBAM(512, 16),features[29])
        self.classify = nn.Sequential(
            nn.Linear(n1_nodes,n2_nodes),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(n2_nodes,1)
        )
    
    def forward(self,x):
        x = x.view(-1, x.size(-3), x.size(-2), x.size(-1))
        x1 = self.features1(x)
        x2 = self.conv2d(x)
        x2 = self.features2(x2)
        x11 = x1.view(x1.size(0),512,-1)
        x22 = x2.view(x2.size(0),-1,512)    
        x2 = F.softmax(F.max_pool2d(x2,2,padding=1),dim=1)
        x_1 = (1+x2)*x1           
        x_1 = F.avg_pool2d(x_1,7)
        x = torch.matmul(x22,x11)
        x = F.adaptive_avg_pool2d(x,(16,32))
        x = x.view(x.size(0),-1)
        x_1 = x_1.view(x_1.size(0),-1)
        x = torch.cat((x,x_1),dim = 1)         
        out = self.classify(x)
        return out, 

class ResnetIQA(nn.Module):
    def __init__(self,weight_file = None,feature_channels = 512,n1_nodes=512,n2_nodes=512):
        super(ResnetIQA,self).__init__()
        resnet = resnet34(pretrained=False)
        if not(weight_file == None):
            resnet.load_state_dict(torch.load(weight_file),strict=False)
        self.features = nn.Sequential(resnet.conv1,resnet.bn1,resnet.relu,
        resnet.maxpool,resnet.layer1,resnet.layer2,
        resnet.layer3,resnet.layer4)
        self.classifier = nn.Sequential(
            nn.Linear(2*feature_channels,n1_nodes),
            nn.ReLU(True),
            nn.BatchNorm1d(n1_nodes),
            nn.Linear(n1_nodes,n2_nodes),
            nn.ReLU(True),
            nn.Linear(n2_nodes,1)
        )

    
    def forward(self,x):
        x = x.view(-1, x.size(-3), x.size(-2), x.size(-1))
        h = self.features(x)
        h1 = F.max_pool2d(h, (h.size(-2), h.size(-1)))
        h2 = -F.max_pool2d(-h, (h.size(-2), h.size(-1)))
        h = torch.cat((h1, h2), 1)  #
        h = h.squeeze(3).squeeze(2)
        out = self.classifier(h)
        return out
