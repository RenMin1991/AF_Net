# -*- coding: utf-8 -*-
'''
    maxout + NetVLAD
    Ren Min
    20181122
'''

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

from netvlad import NetVLAD
from layers import ConvOffset2D


class mfm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, type=1):
        super(mfm, self).__init__()
        self.out_channels = out_channels
        if type == 1:
            self.filter = nn.Conv2d(in_channels, 2*out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.filter = nn.Linear(in_channels, 2*out_channels)

    def forward(self, x):
        x = self.filter(x)
        out = torch.split(x, self.out_channels, 1)
        return torch.max(out[0], out[1])

class Maxout_4(nn.Module):
    def __init__(self, num_classes_th, num_classes_la, num_classes_in, num_classes_cs):
        super(Maxout_4, self).__init__()
        self.features = nn.Sequential(
            mfm(1, 48, 9, 1, 0),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            mfm(48, 96, 5, 1, 0),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            mfm(96, 128, 5, 1, 0),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            mfm(128, 192, 4, 1, 0),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            )
        self.fc1 = mfm(5*5*192, 256, type=0)
        self.dropout = nn.Dropout(0.7)
        self.fc2_th = nn.Linear(256, num_classes_th)
        self.fc2_la = nn.Linear(256, num_classes_la)
        #self.fc2_in = nn.Linear(256, num_classes_in)
        self.fc2_cs = nn.Linear(256, num_classes_cs)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.dropout(x)
        out_th = self.fc2_th(x)
        out_la = self.fc2_la(x)
        #out_in = self.fc2_in(x)
        out_cs = self.fc2_cs(x)
        return out_th, out_la, out_cs, x


fc1_pca_weights = torch.load('fc1_pca_weights.pth')
fc1_pca_weights = fc1_pca_weights.float()
    
class Maxout_VLAD(nn.Module):
    def __init__(self, num_classes_th, num_classes_la, num_classes_in, num_classes_cs):
        super(Maxout_VLAD, self).__init__()
        
        self.mfm1 = mfm(1, 48, 9, 1, 0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.offset1 = ConvOffset2D(filters=48)

        self.mfm2 = mfm(48, 96, 5, 1, 0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.offset2 = ConvOffset2D(filters=96)

        self.mfm3 = mfm(96, 128, 5, 1, 0)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.offset3 = ConvOffset2D(filters=128)

        self.mfm4 = mfm(128, 192, 4, 1, 0)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
       
        self.offset4 = ConvOffset2D(filters=192)
 
        self.netvlad = NetVLAD(num_clusters=25, num_ghost=0, dim=192, alpha=50., normalize_input=True, L2_normalize=False)
        
        self.fc1_pca = mfm(25*192, 256, type=0)
        #self.bn = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.7)
        self.fc2_th_pca = nn.Linear(256, num_classes_th)
        self.fc2_la_pca = nn.Linear(256, num_classes_la)
        self.fc2_in = nn.Linear(256, num_classes_in)
        self.fc2_cs_pca = nn.Linear(256, num_classes_cs)

        #self._init_params()

    #def _init_params(self):
        #self.fc1_pca.filter.weight = nn.Parameter(fc1_pca_weights)

    def forward(self, x):
        x = self.mfm1(x)
        x = self.pool1(x)
        x = self.offset1(x)

        x = self.mfm2(x)
        x = self.pool2(x)
        x = self.offset2(x)

        x = self.mfm3(x)
        x = self.pool3(x)
        x = self.offset3(x)

        x = self.mfm4(x)
        x = self.pool4(x)
        x = self.offset4(x)        

        x = self.netvlad(x)
        
        #x = x.view(x.size(0), -1)
        x = self.fc1_pca(x)

        #x = self.bn(x)
        #x = F.normalize(x, p=2, dim=1)

        x = self.dropout(x)
        out_th = self.fc2_th_pca(x)
        out_la = self.fc2_la_pca(x)
        out_in = self.fc2_in(x)
        out_cs = self.fc2_cs_pca(x)
        return out_th, out_la, out_in, out_cs, x



