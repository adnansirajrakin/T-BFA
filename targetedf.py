from __future__ import print_function
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
# datapath for the workstation
dataset_path='/home/adnan/data/pytorch/cifar10'
# datapath for the mac
# dataset_path='/Users/elliot/dataset/cifar10'
import operator
import argparse
parser = argparse.ArgumentParser(description='Training network for image classification',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--target', type=int, default=8, help='target class from')
parser.add_argument('--toattack', type=int, default=1, help='targeted to')
parser.add_argument('--iters', type=int, default=40, help='attack iteration')
parser.add_argument('--type', type=int, default=1, help='type of the attack')
parser.add_argument('--attacksamp', type=int, default=5000, help='No. of test samples usesd from attacked class 5000 means 50 % 10000 means 100 %')
parser.add_argument('--testsamp', type=int, default=500, help='No. of test samples from all other class 500')
parser.add_argument('--fclayer', type=int, default=9, help='No. of test samples from all other class 500')
parser.add_argument('--model', type=str, default='vgg', help='model resnet20/vgg11=  res/vgg')
args = parser.parse_args()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

import matplotlib.pyplot as plt



def validate1(model, data,target, criterion, val_loader, epoch,xn):    
    model.eval()
    test_loss = 0
    correct = 0
    preds=torch.zeros([10000]) 
    with torch.no_grad():
            data, target = data.cuda(), target.cuda()
            output,_ = model(data)
            test_loss += criterion(output, target).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            #print(pred)
            correct += pred.eq(target.view_as(pred)).sum().item()
        
    
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, xn,
        100. * correct /xn))
    
    return (100. * correct /xn)


def validate(model, device, criterion, val_loader, epoch):    
    model.eval()
    test_loss = 0
    correct = 0
    preds=torch.zeros([10000]) 
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.cuda(), target.cuda()
            output,_ = model(data)
            test_loss += criterion(output, target).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            #print(pred,target)
            correct += pred.eq(target.view_as(pred)).sum().item()
        
    
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, val_loader.sampler.__len__(),
        100. * correct / val_loader.sampler.__len__() ))
    
    return test_loss, 100. * correct / val_loader.sampler.__len__()


def validate2(model, device, criterion, val_loader, epoch):    
    model.eval()
    test_loss = 0
    correct = 0
    preds=torch.zeros([10000]) 
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.cuda(), target.cuda()
            target[:]=args.toattack
            output,_ = model(data)
            test_loss += criterion(output, target).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            #print(pred,target)
            correct += pred.eq(target.view_as(pred)).sum().item()
        
    
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, val_loader.sampler.__len__(),
        100. * correct / val_loader.sampler.__len__() ))
    
    return test_loss, 100. * correct / val_loader.sampler.__len__()


def validate3(model, device, criterion, data1,target1, epoch):    
    model.eval()
    test_loss = 0
    correct = 0
    n=0
    m=100
    preds=torch.zeros([10000]) 
    with torch.no_grad():
        for i in range((9000-args.testsamp)//100):
            data, target = data1[n:m,:,:,:].cuda(), target1[n:m].cuda()
            m+=100
            n+=100
            output,_ = model(data)
            test_loss += criterion(output, target).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            #print(pred,target)
            correct += pred.eq(target.view_as(pred)).sum().item()
        
    
    test_loss /= 9000-args.testsamp
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, (9000-args.testsamp) ,
        100. * correct / (9000-args.testsamp)  ))
    
    return test_loss, 100. * correct / (9000-args.testsamp)    

from torch.autograd import Variable
from torchvision import models
import torch.nn.functional as F

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
        



import torch.nn as nn
import torch.nn.init as init

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

model1=vgg11()

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




model2 = CifarResNet(ResNetBasicblock, 20, 10)

if args.model=='res':
   model=model2
if args.model=='vgg':
   model=model1   
BATCH_SIZE = 128
TEST_BATCH_SIZE = 128
EPOCHS = 100
LR = 0.0005
MOMENTUM=0.9
WEIGHT_DECAY=0.0001
SEED = 1
LOG_INTERVAL = 100



device=0
mean = [x / 255 for x in [129.3, 124.1, 112.4]]
std = [x / 255 for x in [68.2, 65.4, 70.4]]
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((mean), (std)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((mean), (std)),
])

import copy
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train) 

train_loader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=2) 

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test) 
test_loader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=2) 



criterion = torch.nn.CrossEntropyLoss()

criterion=criterion.cuda()


def int2bin(input, num_bits):
    '''
    convert the signed integer value into unsigned integer (2's complement equivalently).
    '''
    output = input.clone()
    output[input.lt(0)] = 2**num_bits + output[input.lt(0)]
  
    return output


def bin2int(input, num_bits):
    '''
    convert the unsigned integer (2's complement equivantly) back to the signed integer format
    with the bitwise operations. Note that, in order to perform the bitwise operation, the input
    tensor has to be in the integer format.
    '''
    mask = 2**(num_bits-1) - 1
    output = -(input & ~mask) + (input & mask)
    return output


def weight_conversion(model):
    '''
    Perform the weight data type conversion between:
        signed integer <==> two's complement (unsigned integer)

    Note that, the data type conversion chosen is depend on the bits:
        N_bits <= 8   .char()   --> torch.CharTensor(), 8-bit signed integer
        N_bits <= 16  .short()  --> torch.shortTensor(), 16 bit signed integer
        N_bits <= 32  .int()    --> torch.IntTensor(), 32 bit signed integer
    '''
    for m in model.modules():
        if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
            w_bin = int2bin(m.weight.data, m.N_bits).char()
            
            m.weight.data = bin2int(w_bin, m.N_bits).float()
    return



def converted(model):
	# model.load_state_dict(torch.load('./cifar_vgg_pretrain.pt', map_location='cpu'))
	pretrained_dict = torch.load('Resnet20_8_0.pkl',map_location=lambda storage, loc: storage)	
	model_dict = model.state_dict()

	# 1. filter out unnecessary keys
	pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
	# 2. overwrite entries in the existing state dict		
	model_dict.update(pretrained_dict) 	
	# 3. load the new state dict	
	model.load_state_dict(model_dict)
	# update the step size before validation
	for m in model.modules():
         if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
        	 m.__reset_stepsize__()
        	 m.__reset_weight__()
	weight_conversion(model)
	return model
def converted1(model):
	# model.load_state_dict(torch.load('./cifar_vgg_pretrain.pt', map_location='cpu'))
	pretrained_dict = torch.load('vgg_8_0.pkl',map_location=lambda storage, loc: storage)	
	model_dict = model.state_dict()

	# 1. filter out unnecessary keys
	pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
	# 2. overwrite entries in the existing state dict		
	model_dict.update(pretrained_dict) 	
	# 3. load the new state dict	
	model.load_state_dict(model_dict)
	# update the step size before validation
	for m in model.modules():
         if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
        	 m.__reset_stepsize__()
        	 m.__reset_weight__()
	weight_conversion(model)
	return model    
    
class BFA1(object):
    def __init__(self, criterion, fc, k_top=10):

        self.criterion = criterion
        # init a loss_dict to log the loss w.r.t each layer
        self.loss_dict = {}
        self.bit_counter = 0
        self.k_top = k_top
        self.n_bits2flip = 0
        self.loss = 0
        self.fc=fc  
    def flip_bit(self, m,offs):
        '''
        the data type of input param is 32-bit floating, then return the data should
        be in the same data_type.
        '''
        self.k_top=  m.weight.grad.detach().abs().view(-1).size()[0]
        # 1. flatten the gradient tensor to perform topk
        w_grad_topk, w_idx_topk = m.weight.grad.detach().abs().view(-1).topk(
            self.k_top)
        # update the b_grad to its signed representation
        w_grad_topk = m.weight.grad.detach().view(-1)[w_idx_topk]

        # 2. create the b_grad matrix in shape of [N_bits, k_top]
        b_grad_topk = w_grad_topk * m.b_w.data

        # 3. generate the gradient mask to zero-out the bit-gradient
        # which can not be flipped
        b_grad_topk_sign = (b_grad_topk.sign() +
                            1) * 0.5  # zero -> negative, one -> positive
        # convert to twos complement into unsigned integer
        w_bin = int2bin(m.weight.detach().view(-1), m.N_bits).short()
        w_bin_topk = w_bin[w_idx_topk]  # get the weights whose grads are topk
        # generate two's complement bit-map
        b_bin_topk = (w_bin_topk.repeat(m.N_bits,1) & m.b_w.abs().repeat(1,self.k_top).short()) \
        // m.b_w.abs().repeat(1,self.k_top).short()
        grad_mask = b_bin_topk ^ b_grad_topk_sign.short()
        
        grad_mask=(torch.ones(grad_mask.size()).short().cuda()-grad_mask).short()
       
        # 4. apply the gradient mask upon ```b_grad_topk``` and in-place update it
        b_grad_topk *= grad_mask.float()

        # 5. identify the several maximum of absolute bit gradient and return the
        # index, the number of bits to flip is self.n_bits2flip
        grad_max = b_grad_topk.abs().max()
        _, b_grad_max_idx = b_grad_topk.abs().view(-1).topk(self.n_bits2flip)
        bit2flip = b_grad_topk.clone().view(-1).zero_()
        
        if grad_max.item() != 0:  # ensure the max grad is not zero
            bit2flip[b_grad_max_idx] = 1
            bit2flip = bit2flip.view(b_grad_topk.size())
        else:
            pass

   
#         print(bit2flip)

# 6. Based on the identified bit indexed by ```bit2flip```, generate another
# mask, then perform the bitwise xor operation to realize the bit-flip.
        w_bin_topk_flipped = (bit2flip.short() * m.b_w.abs().short()).sum(0, dtype=torch.int16) \
                ^ w_bin_topk
        weight_changed=w_bin_topk-w_bin_topk_flipped
        idx=(weight_changed!=0).nonzero() ## index of the weight  changed  
        # 7. update the weight in the original weight tensor
        w_bin[w_idx_topk] = w_bin_topk_flipped  # in-place change
        param_flipped = bin2int(w_bin,
                                m.N_bits).view(m.weight.data.size()).float()
        offse=(w_idx_topk[idx]) 
        return param_flipped,offse


    def progressive_bit_search(self, model, data, target,data1,target1):
        ''' 
        Given the model, base on the current given data and target, go through
        all the layer and identify the bits to be flipped. 
        '''
        # Note that, attack has to be done in evaluation model due to batch-norm.
        # see: https://discuss.pytorch.org/t/what-does-model-eval-do-for-batchnorm-layer/7146
        model.eval()
        #target[:]=3
        # 1. perform the inference w.r.t given data and target
        output,_ = model(data.cuda())
        #         _, target = output.data.max(1)
        self.loss = self.criterion(output, target.cuda())
        # 2. zero out the grads first, then get the grads
        for m in model.modules():
            if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
                if m.weight.grad is not None:
                    m.weight.grad.data.zero_()

        self.loss.backward()
        # init the loss_max to enable the while loop
        self.loss_max = self.loss.item()

        # 3. for each layer flip #bits = self.bits2flip
        
        for j in range(1): 
            self.n_bits2flip += 1
            # iterate all the quantized conv and linear layer
            n=0
            offs=0
            for name, module in model.named_modules():
                if isinstance(module, quan_Conv2d) or isinstance(
                        module, quan_Linear):
                    n=n+1
                    if n<22:
                    #print(n,name)
                        clean_weight = module.weight.data.detach()
                        attack_weight,_ = self.flip_bit(module,offs)
                    # change the weight to attacked weight and get loss
                        module.weight.data = attack_weight
                        output,_ = model(data)
                        self.loss_dict[name] = self.criterion(output,target).item()
                        
                        
                    # change the weight back to the clean weight
                        module.weight.data = clean_weight
                    if n<self.fc:
                        w=module.weight.size()
                        offs+=w[0]*w[1]*w[2]*w[3]  ## keeping track of the offset 
                    else:
                        w=module.weight.size()
                        offs+=w[0]*w[1]   

            # after going through all the layer, now we find the layer with max loss
            max_loss_module = min(self.loss_dict.items(),
                                  key=operator.itemgetter(1))[0]
            self.loss_max = self.loss_dict[max_loss_module]

        # 4. if the loss_max does lead to the degradation compared to the self.loss,
        # then change the that layer's weight without putting back the clean weight
        n=0
        offs=0
        for name, module in model.named_modules():
            if isinstance(module, quan_Conv2d) or isinstance(
                        module, quan_Linear):
                n=n+1
                #print(n,name)
                if name == max_loss_module:
                #print(n,name)
                #                 print(name, self.loss.item(), loss_max)
                    attack_weight,offset = self.flip_bit(module,offs)
                    module.weight.data = attack_weight
                    nn=n
                    print(n,offset)    
                if n<self.fc:
                    w=module.weight.size()
                    offs+=w[0]*w[1]*w[2]*w[3]  ## keeping track of the offset
 
                else:
                    w=module.weight.size()
                    offs+=w[0]*w[1]
                        
        # reset the bits2flip back to 0
        self.bit_counter += self.n_bits2flip
        self.n_bits2flip = 0

        return  nn, offset     

    def progressive_bit_search1(self, model, data, target,data1,target1):
        ''' 
        Given the model, base on the current given data and target, go through
        all the layer and identify the bits to be flipped. 
        '''
        # Note that, attack has to be done in evaluation model due to batch-norm.
        # see: https://discuss.pytorch.org/t/what-does-model-eval-do-for-batchnorm-layer/7146
        model.eval()
        #target[:]=3
        # 1. perform the inference w.r.t given data and target
        output,_ = model(data.cuda())
        
        #         _, target = output.data.max(1)
        self.loss = self.criterion(output, target.cuda())
        output1,_ = model(data1)
        self.loss +=self.criterion(output1,target1).item()
        # 2. zero out the grads first, then get the grads
        for m in model.modules():
            if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
                if m.weight.grad is not None:
                    m.weight.grad.data.zero_()

        self.loss.backward()
        # init the loss_max to enable the while loop
        self.loss_max = self.loss.item()

        # 3. for each layer flip #bits = self.bits2flip
        
        for j in range(1): 
            self.n_bits2flip += 1
            # iterate all the quantized conv and linear layer
            n=0
            offs=0
            for name, module in model.named_modules():
                if isinstance(module, quan_Conv2d) or isinstance(
                        module, quan_Linear):
                    n=n+1
                    if n<22:
                    #print(n,name)
                        clean_weight = module.weight.data.detach()
                        attack_weight,_ = self.flip_bit(module,offs)
                    # change the weight to attacked weight and get loss
                        module.weight.data = attack_weight
                        output,_ = model(data)
                        self.loss_dict[name] = self.criterion(output,target).item()
                        output1,_ = model(data1)
                        xx=self.criterion(output1,target1).item()
                        print(xx,self.loss_dict[name])
                        self.loss_dict[name]+=xx
                        
                    # change the weight back to the clean weight
                        module.weight.data = clean_weight
                    if n<self.fc:
                        w=module.weight.size()
                        offs+=w[0]*w[1]*w[2]*w[3]  ## keeping track of the offset 
                    else:
                        w=module.weight.size()
                        offs+=w[0]*w[1]   

            # after going through all the layer, now we find the layer with max loss
            max_loss_module = min(self.loss_dict.items(),
                                  key=operator.itemgetter(1))[0]
            self.loss_max = self.loss_dict[max_loss_module]

        # 4. if the loss_max does lead to the degradation compared to the self.loss,
        # then change the that layer's weight without putting back the clean weight
        n=0
        offs=0
        for name, module in model.named_modules():
            if isinstance(module, quan_Conv2d) or isinstance(
                        module, quan_Linear):
                n=n+1
                #print(n,name)
                if name == max_loss_module:
                #print(n,name)
                #                 print(name, self.loss.item(), loss_max)
                    attack_weight,offset = self.flip_bit(module,offs)
                    module.weight.data = attack_weight
                    print(n,offset)    
                if n<self.fc:
                    w=module.weight.size()
                    offs+=w[0]*w[1]*w[2]*w[3]  ## keeping track of the offset
 
                else:
                    w=module.weight.size()
                    offs+=w[0]*w[1]
                        
        # reset the bits2flip back to 0
        self.bit_counter += self.n_bits2flip
        self.n_bits2flip = 0

        return  n, offset  



# see: https://discuss.pytorch.org/t/what-does-model-eval-do-for-batchnorm-layer/7146
model.eval()

epoch=1
rounds=1
numbers=torch.zeros([rounds])
values=torch.zeros([rounds])
acc=torch.Tensor(rounds,epoch+1).fill_(0)
if args.model=='res':
    converted(model)
if args.model=='vgg':
    converted1(model)
model=model.cuda()

validate(model, test_loader, criterion, test_loader, 0)
# test case 2


def data_gen():
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True, num_workers=2) 
    target_c=args.target
    tothe=args.toattack

             


    ## dataT and targetT will contain only one class images whose image will be miss-classified
    data=torch.zeros([9000,3,32,32]).cuda()  ## rest of the data 
    target=torch.zeros([9000]).long().cuda() 
    dataT=torch.zeros([1000,3,32,32]).cuda()  ## all the target class images
    targetT=torch.zeros([1000]).long().cuda()
    xs=0
    xn=0
    for t, (x, y) in enumerate (test_loader):
        if t<10000:
            
            if y!=target_c:
               data[xs,:,:,:] = x[0,:,:,:]
               target[xs]=y.long()
               xs+=1
            if y==target_c:
               dataT[xn,:,:,:] = x[0,:,:,:]
               targetT[xn]=y.long()
               xn+=1


    print(xs,xn)
                

    ##dataT1 and targetT1 contains all the images of that target class helps in evaluation

    data1=torch.zeros([args.testsamp,3,32,32]).cuda()
    target1=torch.zeros([args.testsamp]).long().cuda()
    data1=torch.zeros([9000-args.testsamp,3,32,32]).cuda()
    target1=torch.zeros([9000-args.testsamp]).long().cuda()
    dataT1=torch.zeros([args.attacksamp,3,32,32]).cuda()
    targetT1=torch.zeros([args.attacksamp]).long().cuda()
    dataT2=torch.zeros([args.attacksamp,3,32,32]).cuda()
    targetT2=torch.zeros([args.attacksamp]).long().cuda()
    xss=0
    xnn=0



           
    data1=data[0:args.testsamp,:,:,:]  ## only seperating test samples
    target1=target[0:args.testsamp]
    data2=data[args.testsamp:,:,:,:] ## only separating other samples  that is not used for attack
    target2=target[args.testsamp:]
    dataT1=dataT[0:args.attacksamp,:,:,:]  ## seperating attack samples
    targetT1=targetT[0:args.attacksamp]
    dataT2=dataT[1000-args.attacksamp:1000,:,:,:] ## seperating all the rest of the sample from target class
    targetT2=targetT[1000-args.attacksamp:1000]

    test_loader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=True, num_workers=2)
    ##  random test batch for type 1 attack
    for t, (x, y) in enumerate (test_loader):
        xx,yy= x.cuda(),y.cuda()
        break
    yy[:]=tothe
    return data,target,data1,target1,data2,target2,dataT,targetT,dataT1,targetT1,dataT2,targetT2,xx,yy
rounds=args.iters
avgs=5
target_c=args.target
tothe=args.toattack
acc=torch.Tensor(avgs,rounds+1).fill_(0) ## accuracy track
acc1=torch.Tensor(avgs,rounds+1).fill_(0) ## accuracy without attacked class and without test samples used for attack
temp=torch.Tensor(avgs,rounds+1).fill_(0) ##ASR on attack samples
temp1=torch.Tensor(avgs,rounds+1).fill_(0) ## ASR on rest of the  sample
layer=torch.Tensor(avgs,rounds+1).fill_(0) 
offsets=torch.Tensor(avgs,rounds+1).fill_(0)
bfas=torch.Tensor(avgs).fill_(0) ## recording number of bitflips
attacker1 = BFA1(criterion,args.fclayer)
## type 1 attack
if args.type ==1:
    for j in range(avgs):
        if args.model=='res':
            converted(model)
        if args.model=='vgg':
            converted1(model)
        model=model.cuda()
        model.eval()
        data,target,data1,target1,data2,target2,dataT,targetT,dataT1,targetT1,dataT2,targetT2,xx,yy=data_gen()
        _,acc[j,0]=validate(model, device, criterion, test_loader, 0)
        _,temp[j,0]=validate2(model, device, criterion, test_loader, 0)
        for r in range(rounds):
            layer[j,r+1],offsets[j,r+1]=attacker1.progressive_bit_search(model, xx, yy,data1,target1.long())
            print(r+1)
            _,acc[j,r+1]=validate(model, device, criterion, test_loader, 0)
            _,temp[j,r+1]=validate2(model, device, criterion, test_loader, 0)
        
            if temp[j,r+1]>99.99:
                break
        bfas[j]=int(r+1)     
    

## type 2 attack
if args.type ==2:
    for j in range(avgs):
        if args.model=='res':
            converted(model)
        if args.model=='vgg':
            converted1(model)
        model=model.cuda()
        model.eval()
        data,target,data1,target1,data2,target2,dataT,targetT,dataT1,targetT1,dataT2,targetT2,xx,yy=data_gen()

        targetT[:]=tothe
        targetT1[:]=tothe
        targetT2[:]=tothe
        _,acc[j,0]=validate(model, device, criterion, test_loader, 0)
        _,acc1[j,0]=validate3(model, device, criterion, data2,target2, 0)
        temp[j,0]=validate1(model, dataT1,targetT1.long(), criterion, test_loader, 0,args.attacksamp)
        temp1[j,0]=validate1(model, dataT2,targetT2.long(), criterion, test_loader, 0,args.attacksamp)
    
        for r in range(rounds):
            layer[j,r+1],offsets[r+1]=attacker1.progressive_bit_search(model, dataT1, targetT1.long(),data1,target1.long())
            print(r+1)
            _,acc[j,r+1]=validate(model, device, criterion, test_loader, 0)
            _,acc1[j,r+1]=validate3(model, device, criterion, data2,target2, 0)
            temp[j,r+1]=validate1(model, dataT1,targetT1.long(), criterion, test_loader, 0,args.attacksamp)
        
            temp1[j,r+1]=validate1(model, dataT2,targetT2.long(), criterion, test_loader, 0,args.attacksamp)
            if float(temp1[j,r+1])> 99.99:
                break
        bfas[j]=int(r+1)    
        

## type 3 attack
if args.type ==3:
    for j in range(avgs):
        if args.model=='res':
            converted(model)
        if args.model=='vgg':
            converted1(model)
        model=model.cuda()
        data,target,data1,target1,data2,target2,dataT,targetT,dataT1,targetT1,dataT2,targetT2,xx,yy=data_gen()
        targetT[:]=tothe
        targetT1[:]=tothe
        targetT2[:]=tothe
        _,acc[j,0]=validate(model, device, criterion, test_loader, 0)
        _,acc1[j,0]=validate3(model, device, criterion, data2,target2, 0)
        temp[j,0]=validate1(model, dataT1,targetT1.long(), criterion, test_loader, 0,args.attacksamp)
        temp1[j,0]=validate1(model, dataT2,targetT2.long(), criterion, test_loader, 0,args.attacksamp)
        for r in range(rounds):
            layer[j,r+1],offsets[j,r+1]=attacker1.progressive_bit_search1(model, dataT1, targetT1.long(),data1,target1.long())
            print(r+1)
            _,acc[j,r+1]=validate(model, device, criterion, test_loader, 0)
            _,acc1[j,r+1]=validate3(model, device, criterion, data2,target2, 0)
            temp[j,r+1]=validate1(model, dataT1,targetT1.long(), criterion, test_loader, 0,args.attacksamp)
            temp1[j,r+1]=validate1(model, dataT2,targetT2.long(), criterion, test_loader, 0,args.attacksamp)
            if float(temp1[j,r+1])> 99.9:
                break
            if r>1:
                if temp1[j,r-1]==temp1[j,r+1]:
                    break
        bfas[j]=int(r+1) 

test_acc=torch.Tensor(avgs).fill_(0) ## recording test accuracy   
ASR_as=torch.Tensor(avgs).fill_(0) ## recording ASR
ASR_val=torch.Tensor(avgs).fill_(0) ## recording validation ASR
rem_acc=torch.Tensor(avgs).fill_(0) ## recording accuracy on ramaining data    

for i in range(avgs):
    test_acc[i]= acc[i,int(bfas[i])]
    ASR_as[i]= temp[i,int(bfas[i])]
    ASR_val[i]= temp1[i,int(bfas[i])]
    rem_acc[i]= acc1[i,int(bfas[i])]
   

from xlwt import Workbook
wb = Workbook()
# add_sheet is used to create sheet. 
sheet1 = wb.add_sheet('Sheet1')
sheet1.write(0, 0, ("Accuracy")) 
sheet1.write(0, 1, ("ASR_AS "))
sheet1.write(0, 2, ("ASR_rest "))
sheet1.write(0, 3, ("layer number"))
sheet1.write(0, 4, ("offset"))
sheet1.write(0, 5, ("Accuracy_reform"))
count=0
for j in range(avgs):
    for p in range(int(bfas[j])+1):
        sheet1.write(p+1+count, 0, float(acc[j,p])) 
        sheet1.write(p+1+count, 1, float(temp[j,p]))
        sheet1.write(p+1+count, 2, float(temp1[j,p]))
        sheet1.write(p+1+count, 3, float(layer[j,p]))
        sheet1.write(p+1+count, 4, float(offsets[j,p]))
        sheet1.write(p+1+count, 5, float(acc1[j,p]))
   
    count+= int(bfas[j])+2        

sheet2 = wb.add_sheet('Sheet2')
sheet2.write(0, 0, ("Test_acc"))
sheet2.write(0, 1, ("Ref_Acc")) 
sheet2.write(0, 2, ("ASR_AS "))
sheet2.write(0, 3, ("ASR_rest ")) 
sheet2.write(0, 4, ("Bitflip ")) 
sheet2.write(1, 0, float(test_acc.mean()))
sheet2.write(2, 0, float(test_acc.std()))
sheet2.write(1, 1, float(rem_acc.mean()))
sheet2.write(2, 1, float(rem_acc.std()))
sheet2.write(1, 2, float(ASR_as.mean()))
sheet2.write(2, 2, float(ASR_as.std()))
sheet2.write(1, 3, float(ASR_val.mean()))
sheet2.write(2, 3, float(ASR_val.std()))
sheet2.write(1, 4, float(bfas.mean()))
sheet2.write(2, 4, float(bfas.std()))
wb.save( str(args.model) + "_type" + str(args.type) + "_from"+str(target_c)+ "_to"+str(tothe) + ".xls")
