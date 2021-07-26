from torch.autograd import Variable
from torchvision import models
import torch.nn.functional as F



import torch.nn as nn
import torch.nn.init as init
import numpy as np
import pandas as pd
import torch.nn as nn
import math
import torch.nn.functional as F
import torch
from torch.nn import init
from collections import OrderedDict
import time
import shutil
import xlwt
from xlwt import Workbook 
import torchvision
import argparse
import torch.optim as optim
from torchvision import datasets, transforms
# from utils import AverageMeter, RecorderMeter, time_string, convert_secs2time
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
from xlwt import Workbook 
import torch as th

class _quantize_func(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, step_size, half_lvls):
        # ctx is a context object that can be used to stash information
        # for backward computation
        ctx.step_size = step_size
        ctx.half_lvls = half_lvls
        output = F.hardtanh(input,
                            min_val=-ctx.half_lvls * ctx.step_size.item(),
                            max_val=ctx.half_lvls * ctx.step_size.item())

        output = torch.round(output/ctx.step_size)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()/ctx.step_size

        return grad_input, None, None


quantize = _quantize_func.apply


class quan_Conv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(quan_Conv2d, self).__init__(in_channels, out_channels, kernel_size,
                                          stride=stride, padding=padding, dilation=dilation,
                                          groups=groups, bias=bias)
        self.N_bits = 8
        self.full_lvls = 2**self.N_bits
        self.half_lvls = (self.full_lvls-2)/2
        # Initialize the step size
        self.step_size = nn.Parameter(torch.Tensor([1]), requires_grad=True)
        self.__reset_stepsize__()
        # flag to enable the inference with quantized weight or self.weight
        self.inf_with_weight = False  # disabled by default
        
        # create a vector to identify the weight to each bit
        self.b_w = nn.Parameter(
            2**torch.arange(start=self.N_bits-1,end=-1, step=-1).unsqueeze(-1).float(),
            requires_grad = False)
        
        self.b_w[0] = -self.b_w[0] #in-place change MSB to negative
        

    def forward(self, input):
        if self.inf_with_weight:
            #self.weight.data[self.weight.data.abs()>68]=0
            return F.conv2d(input, self.weight*self.step_size, self.bias, self.stride, self.padding,
                            self.dilation, self.groups)
        else:
            weight_quan = quantize(self.weight, self.step_size,
                                   self.half_lvls)*self.step_size
                                   
            return F.conv2d(input, weight_quan, self.bias, self.stride, self.padding, self.dilation,
                            self.groups)

    def __reset_stepsize__(self):
        with torch.no_grad():
            self.step_size.data = (self.weight.abs().max()*2)/self.half_lvls

    def __reset_weight__(self):
        '''
        This function will reconstruct the weight stored in self.weight.
        Replacing the orginal floating-point with the quantized fix-point
        weight representation.
        '''
        # replace the weight with the quantized version
        with torch.no_grad():
            self.weight.data = quantize(
                self.weight, self.step_size, self.half_lvls)
        # enable the flag, thus now computation does not invovle weight quantization
        self.inf_with_weight = True



class quan_Linear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True):
        super(quan_Linear, self).__init__(in_features, out_features, bias=bias)

        self.N_bits = 8
        self.full_lvls = 2**self.N_bits
        self.half_lvls = (self.full_lvls-2)/2
        # Initialize the step size
        self.step_size = nn.Parameter(torch.Tensor([1]), requires_grad=True)
        self.__reset_stepsize__()
        # flag to enable the inference with quantized weight or self.weight
        self.inf_with_weight = False  # disabled by default
        
        # create a vector to identify the weight to each bit
        self.b_w = nn.Parameter(
            2**torch.arange(start=self.N_bits-1,end=-1, step=-1).unsqueeze(-1).float(),
            requires_grad = False)
        
        self.b_w[0] = -self.b_w[0] #in-place reverse

    def forward(self, input):
        if self.inf_with_weight:
            #self.weight.data[self.weight.data.abs()>68]=0
            return  F.linear(input, self.weight*self.step_size, self.bias)
        else:
             
            weight_quan = quantize(self.weight, self.step_size,
                               self.half_lvls)*self.step_size
            
            return F.linear(input, weight_quan, self.bias)

    def __reset_stepsize__(self):
        with torch.no_grad():
            self.step_size.data = (self.weight.abs().max()*2)/self.half_lvls

    def __reset_weight__(self):
        '''
        This function will reconstruct the weight stored in self.weight.
        Replacing the orginal floating-point with the quantized fix-point
        weight representation.
        '''
        # replace the weight with the quantized version
        with torch.no_grad():
            self.weight.data = quantize(
                self.weight, self.step_size, self.half_lvls)
        # enable the flag, thus now computation does not invovle weight quantization
        self.inf_with_weight = True    




__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


class VGG(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            quan_Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            quan_Linear(512, 512),
            nn.ReLU(True),
            quan_Linear(512, 10),
        )
         # Initialize weights
        for m in self.modules():
            if isinstance(m, quan_Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                


    def forward(self, x,tests=False,calss=False,n=None):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        if tests==True:
            if calss==True:
                for k in range(len(n)):
                    x[0,n[k]]=0 
        out = self.classifier(x)
        return out,x


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = quan_Conv2d(in_channels, v, kernel_size=3,stride=1, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
          512, 512, 512, 512, 'M'],
}


def vgg11(num_classes=10):
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A'],batch_norm=True))

class DownsampleA(nn.Module):

  def __init__(self, nIn, nOut, stride):
    super(DownsampleA, self).__init__()
    assert stride == 2
    self.avg = nn.AvgPool2d(kernel_size=1, stride=stride)

  def forward(self, x):
    x = self.avg(x)
    return torch.cat((x, x.mul(0)), 1)

class ResNetBasicblock(nn.Module):
  expansion = 1
  """
  RexNet basicblock (https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua)
  """
  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(ResNetBasicblock, self).__init__()

    self.conv_a = quan_Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bn_a = nn.BatchNorm2d(planes)

    self.conv_b = quan_Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn_b = nn.BatchNorm2d(planes)

    self.downsample = downsample

  def forward(self, x):
    residual = x

    basicblock = self.conv_a(x)
    basicblock = self.bn_a(basicblock)
    basicblock = F.relu(basicblock, inplace=True)

    basicblock = self.conv_b(basicblock)
    basicblock = self.bn_b(basicblock)

    if self.downsample is not None:
      residual = self.downsample(x)
    
    return F.relu(residual + basicblock, inplace=True)

class CifarResNet(nn.Module):
  """
  ResNet optimized for the Cifar dataset, as specified in
  https://arxiv.org/abs/1512.03385.pdf
  """
  def __init__(self, block, depth, num_classes):
    """ Constructor
    Args:
      depth: number of layers.
      num_classes: number of classes
      base_width: base width
    """
    super(CifarResNet, self).__init__()

    #Model type specifies number of layers for CIFAR-10 and CIFAR-100 model
    assert (depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110'
    layer_blocks = (depth - 2) // 6
    print ('CifarResNet : Depth : {} , Layers for each block : {}'.format(depth, layer_blocks))

    self.num_classes = num_classes
    self.dropout1= nn.Dropout2d(0.3)
    self.conv_1_3x3 = quan_Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn_1 = nn.BatchNorm2d(16)

    self.inplanes = 16
    self.stage_1 = self._make_layer(block, 16, layer_blocks, 1)
    self.stage_2 = self._make_layer(block, 32, layer_blocks, 2)
    self.stage_3 = self._make_layer(block, 64, layer_blocks, 2)
    self.avgpool = nn.AvgPool2d(8)
    self.dropout=nn.Dropout(0.8)
    self.classifier = quan_Linear(64*block.expansion, num_classes)
    

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        #m.bias.data.zero_()
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
      elif isinstance(m, nn.Linear):
        init.kaiming_normal(m.weight)
        m.bias.data.zero_()

  def _make_layer(self, block, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = DownsampleA(self.inplanes, planes * block.expansion, stride)

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(self.inplanes, planes))

    return nn.Sequential(*layers)

  def forward(self, x):
    #x=self.dropout1(x)
    x = self.conv_1_3x3(x)
    x = F.relu(self.bn_1(x), inplace=True)
    x = self.stage_1(x)
    x = self.stage_2(x)
    x = self.stage_3(x)
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    out=self.classifier(x)               
    return out,x