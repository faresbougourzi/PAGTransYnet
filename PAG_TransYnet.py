# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 15:43:00 2024

@author: FaresBougourzi
"""


import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage

import math

from os.path import join as pjoin
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms.functional as TF

import timm


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class StdConv2d(nn.Conv2d):

    def forward(self, x):
        w = self.weight
        # print(w.shape)
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride,
                     padding=1, bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride,
                     padding=0, bias=bias)



##### Double Convs
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
                
        self.conv = nn.Sequential(
            conv3x3(in_channels, out_channels, 1, 1, bias=False),
            nn.GroupNorm(32, out_channels, eps=1e-6),
            nn.ReLU(inplace=True),
            conv3x3(out_channels, out_channels, 1, 1, bias=False),
            nn.GroupNorm(32, out_channels, eps=1e-6),
            nn.ReLU(inplace=True),
        ) 
        self.skip = nn.Sequential(
            conv1x1(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(32, out_channels, eps=1e-6),
            nn.ReLU(inplace=True))           

    def forward(self, x):
        return self.conv(x) + self.skip(x)  

########################################
class DoubleConv2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv2, self).__init__()
                
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ) 
        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))           

    def forward(self, x):
        return self.conv(x) + self.skip(x)     

#### PUNet #########################################################
###### Attention Block
class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)
        return x*psi
    
    
#################################################

###### Attention Block
class Attention_gate(nn.Module):
    def __init__(self,F_p, F_m, F_v, F_int):
        super(Attention_gate, self).__init__()
        self.W_g1 = nn.Sequential(
            nn.Conv2d(F_p, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x1 = nn.Sequential(
            nn.Conv2d(F_m, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_g2 = nn.Sequential(
            nn.Conv2d(F_m, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )  
        
        self.W_x2 = nn.Sequential(
            nn.Conv2d(F_v, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi1 = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.psi2 = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )        
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, p, m, v):
        g1 = self.W_g1(p)
        x1 = self.W_x1(m)
        g2 = self.W_g2(m)
        x2 = self.W_x2(v)
        psi1 = self.relu(g1+x1)
        psi1 = self.psi1(psi1)
        psi2 = self.relu(g2+x2)
        psi2 = self.psi2(psi2)        
        out = torch.cat((m*psi1, v*psi2), dim=1)
        return out
    
    
#################################################
  

class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)
        bn = nn.BatchNorm2d(out_channels)
        super(Conv2dReLU, self).__init__(conv, bn, relu) 
        
        
#### PAGTransYnet #########################################################  
class PAGTransYnet(nn.Module):
    def __init__(self, input_channels=3, num_classes = 1, deep_supervision=False):
        super(PAGTransYnet, self).__init__()
        
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)         

        
        nb_filter = [32, 64, 128, 256, 512]
        nb_filter2 = [input_channels, 32, 64, 128, 256, 512]
        conv_nb = [1, 2, 3, 4]
        self.nb_filter = nb_filter

        self.conv0 = DoubleConv(input_channels, nb_filter[0])
        self.conv1 = DoubleConv(nb_filter[0]+nb_filter[1], nb_filter[1])
        self.conv2 = DoubleConv(nb_filter[1]+nb_filter[2], nb_filter[2])
        self.conv3 = DoubleConv(nb_filter[2]+nb_filter[3]+nb_filter[1], nb_filter[3])
        self.conv4 = DoubleConv(nb_filter[3]+nb_filter[4], nb_filter[4])
        self.conv5 = DoubleConv(768, nb_filter[4]*2)
        
        convp1 =   nn.ModuleList(
                    [
                    DoubleConv(nb_filter2[i], nb_filter2[i+1])                    
                        for i in range(conv_nb[0])
                    ])
        self.convp1 = nn.Sequential(*convp1)
        
        convp2 =  nn.ModuleList(
                    [
                    DoubleConv(nb_filter2[i], nb_filter2[i+1])                    
                        for i in range(conv_nb[1])
                    ])
        self.convp2 = nn.Sequential(*convp2)
        
        convp3 =  nn.ModuleList(
                    [
                    DoubleConv(nb_filter2[i], nb_filter2[i+1])                    
                        for i in range(conv_nb[2])
                    ])
        self.convp3 = nn.Sequential(*convp3)
        
        convp4 =  nn.ModuleList(
                    [
                    DoubleConv(nb_filter2[i], nb_filter2[i+1])                    
                        for i in range(conv_nb[3])
                    ])
        self.convp4 = nn.Sequential(*convp4)
    
        
        self.Attg1 = Attention_gate(F_p=nb_filter[0], F_m=nb_filter[0], F_v=64, F_int= int(nb_filter[0]/2))
        self.Attg2 = Attention_gate(F_p=nb_filter[1], F_m=nb_filter[1], F_v=128, F_int=nb_filter[0])
        self.Attg3 = Attention_gate(F_p=nb_filter[2], F_m=nb_filter[2], F_v=320, F_int=nb_filter[1])
        self.Attg4 = Attention_gate(F_p=nb_filter[3], F_m=nb_filter[3], F_v=512, F_int=nb_filter[2])
        
        
        self.transformer = timm.create_model(
            'pvt_v2_b2_li',
            pretrained=True,
            features_only=True,
        )
        
        self.transformer4 = timm.create_model('vit_base_r50_s16_224.orig_in21k', pretrained=True)
        self.patch_embed = self.transformer4.patch_embed.proj  
        self.transformer4 = self.transformer4.blocks       
        
        self.conv_more1 = Conv2dReLU(
            768,
            512,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        ) 
                      
        self.conv3_1 = DoubleConv2(3*nb_filter[3]+nb_filter[4], nb_filter[3])
        self.conv2_2 = DoubleConv2(3*nb_filter[2]+nb_filter[3], nb_filter[2])
        self.conv1_3 = DoubleConv2(3*nb_filter[1]+nb_filter[2], nb_filter[1])
        self.conv0_4 = DoubleConv2(3*nb_filter[0]+nb_filter[1], nb_filter[0])          
        
        
        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        
    def tr_reshape(self, x):
        B, n_patch, hidden = x.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))        
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)        
        return x
        

        
    def forward(self, input):
        # Images
        input = input.repeat(1, 3, 1, 1)
        image_size = input.shape
        Images = []
        divsize = [2, 4, 8, 16]
        for i in range(len(self.nb_filter)-1):
            Images.append(TF.resize(input, size=[int(image_size[2]/divsize[i]) , int(image_size[3]/divsize[i])]))
        
        
        '''Encoder'''
        # x0 ---> x0
        x0 = self.conv0(input)
        trfea = self.transformer(input)
        
        # x1, tr1
        x1 = self.convp1(Images[0])
        x1 = self.Attg1(p=x1, m=self.pool(x0), v=self.up(trfea[0]))        
        x1 = self.conv1(x1)         

        
        # x2, tr2
        x2 = self.convp2(Images[1])
        x2 = self.Attg2(p=x2, m=self.pool(x1), v=self.up(trfea[1]))        
        x2 = self.conv2(x2)  
        
        # x3, tr3
        x3 = self.convp3(Images[2])       
        x3 = self.Attg3(p=x3, m=self.pool(x2), v=self.up(trfea[2]))        
        x3 = self.conv3(x3)
        
        # x4, tr4
        x4 = self.convp4(Images[3])
        x4 = self.Attg4(p=x4, m=self.pool(x3), v=self.up(trfea[3]))        
        x4 = self.conv4(x4)
        
        x41 = self.conv5(torch.cat((x4, self.pool(x3)), dim=1))

        # tr4 
        x41 = self.patch_embed(x41)
        x41 = x41.flatten(2).transpose(1, 2)
        tr4 = self.transformer4(x41)
        tr4 = self.tr_reshape(tr4)        
        x5 = self.conv_more1(tr4)
        
        
        
        
        '''Decoder'''
        d4 = self.up(x5)
        x4 = torch.cat((x3, self.up(x4)), dim=1)
        d3 = self.conv3_1(torch.cat((x4, d4), dim=1))
        
        d2 = self.up(d3)
        x3 = torch.cat((x2, self.up(x3)), dim=1)
        d2 = self.conv2_2(torch.cat((x3, d2),dim=1)) 
                
        d1 = self.up(d2)
        x2 = torch.cat((x1, self.up(x2)), dim=1)
        d1 = self.conv1_3(torch.cat((x2, d1),dim=1)) 

        d0 = self.up(d1)
        x1 = torch.cat((x0, self.up(x1)), dim=1)
        d0 = self.conv0_4(torch.cat((x1, d0),dim=1))                
        
        output = self.final(d0)         

        return output 

    
#######################

net = PAGTransYnet(input_channels=3, num_classes= 9)

inp = torch.rand(1,1,224,224)
out = net(inp)

print(out.shape)


