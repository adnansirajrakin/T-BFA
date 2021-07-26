##------------------ modules  -------------------------

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
from attack import BFA1,int2bin,bin2int,weight_conversion
from module import vgg11,CifarResNet,ResNetBasicblock
from module import quan_Conv2d,quan_Linear
import operator
import argparse
import copy
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Usign T-BFA',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--target', type=int, default=8, help='target class from')
parser.add_argument('--toattack', type=int, default=1, help='targeted to')
parser.add_argument('--iters', type=int, default=40, help='attack iteration')
parser.add_argument('--type', type=int, default=1, help='type of the attack')
parser.add_argument('--attacksamp', type=int, default=500, help='No. of test samples usesd from attacked class 5000 means 50 % 10000 means 100 %')
parser.add_argument('--testsamp', type=int, default=500, help='No. of test samples from all other class 500')
parser.add_argument('--fclayer', type=int, default=20, help='No. of test samples from all other class 500')
parser.add_argument('--model', type=str, default='res', help='model resnet20/vgg11=  res/vgg')
args = parser.parse_args()





def validate1(model, data,target, criterion, val_loader, epoch,xn):

    "this function computes the accuracy for a given data and target on model"    
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
    print('\nSubTest set: Average loss: {:.4f}, Attack Success Rate: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, xn,
        100. * correct /xn))
    
    return (100. * correct /xn)


def validate(model, device, criterion, val_loader, epoch):  
    "test function" 
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
    "this function computes the attack success rate of  all data to target class toattack"    
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
    print('\nTest set: loss: {:.4f}, Attack success rate: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, val_loader.sampler.__len__(),
        100. * correct / val_loader.sampler.__len__() ))
    
    return test_loss, 100. * correct / val_loader.sampler.__len__()


def validate3(model, device, criterion, data1,target1, epoch):   
    "this function computes the accuracy of a given data1 and target1 batchwise"
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
    print('\nSub Test set: loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, (9000-args.testsamp) ,
        100. * correct / (9000-args.testsamp)  ))
    
    return test_loss, 100. * correct / (9000-args.testsamp)    


        

# ----------------------- model definition -----------------------------

model1=vgg11()
model2 = CifarResNet(ResNetBasicblock, 20, 10)
if args.model=='res':
   model=model2
if args.model=='vgg':
   model=model1   

###----------------------------------- Data prep -----------------------------
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


trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train) 

train_loader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=2) 

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test) 
test_loader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=2) 



criterion = torch.nn.CrossEntropyLoss()
criterion=criterion.cuda()



## -------------------------- loading the models ------------------------------------
def converted(model):
	# model.load_state_dict(torch.load('./cifar_vgg_pretrain.pt', map_location='cpu'))
	pretrained_dict = torch.load('./models/Resnet20_8_0.pkl',map_location=lambda storage, loc: storage)	
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
	pretrained_dict = torch.load('./models/vgg_8_0.pkl',map_location=lambda storage, loc: storage)	
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
    




# see: https://discuss.pytorch.org/t/what-does-model-eval-do-for-batchnorm-layer/7146
model.eval()


## hyper parameters
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


##------------------------------------ generating the data for attack ------------------------ 
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


# attack record
rounds=args.iters # attakc iterations
avgs=5 # average of fice rounds
target_c=args.target # source class
tothe=args.toattack # target class
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
wb.save( "./result/" + str(args.model) + "_type" + str(args.type) + "_from"+str(target_c)+ "_to"+str(tothe) + ".xls")
