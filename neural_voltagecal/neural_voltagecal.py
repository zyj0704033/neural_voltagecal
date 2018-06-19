# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import utils
from network import FCnet,Convnet,FCConvnet
import time
import matplotlib.pyplot as plt
import os
import numpy as np
import math


learning_rate = 0.0005
learning_rate_decay = 0
momentum = 0.9
epoch_num = 100
log_num = 400
batch_size = 8
arch = 'Convnet'
if arch == None:
    arch = 'FCnet'

def weight_init(m):
    #TODO:More initialzation method
    print(m.__class__.__name__)
    if isinstance(m,nn.Linear):
        print(m.in_features)
        torch.nn.init.constant(m.weight.data,1/m.in_features)
        torch.nn.init.constant(m.bias.data,0)
    if isinstance(m,nn.Conv1d):
        print(m.kernel_size)
        n = m.kernel_size[0]*m.out_channels
        m.weight.data.normal_(0,math.sqrt(2./n))
    if isinstance(m,nn.Conv2d):
        n = m.kernel_size[0]*m.kernel_size[1]*m.out_channels
        m.weight.data.normal_(0,math.sqrt(2./n))
    if isinstance(m,nn.BatchNorm1d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


def train():
    dataset = utils.Mydataset()
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=True)
    if arch=='Convnet':
        net = Convnet()
    elif arch=='FCConvnet':
        net = FCConvnet()
    else:
        net = FCnet()
    print(net)
    net.apply(weight_init)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(),lr=learning_rate,weight_decay=learning_rate_decay)
    loss_list = []
    print('\033[1;32m start to train the network!\033[0m')
    for epoch in range(epoch_num):
        
        '''
         Train the network
        '''
        net.train()
        for i,data in enumerate(dataloader):
            P = data[0]
            Q = data[1]
            V = data[2]
            if arch=='Convnet':
                Q = Q.view(batch_size,1,-1)
            Q,V = Variable(Q),Variable(V)
            optimizer.zero_grad()
            Vout = net(Q)
            loss =  criterion(Vout,V)
            loss_list.append(loss.data[0])
            loss.backward()
            optimizer.step()
            if i%log_num==0:
                print('\033[1;33m epoch number: \033[0m%d;  \033[1;33m iteration number: \033[0m%d;  \
                \033[1;33m loss: \033[0m%f'%(epoch,i,loss.data[0]))
                print(Vout.data[0].numpy())
                print(V.data[0].numpy())
        
    #save model
    save_model_path = '../data/model/'+arch+'_'+'epoch_'+str(epoch_num)+'_'+str(time.ctime()).replace(' ','_')+'.model'
    torch.save(net.state_dict(),save_model_path)
    print('\033[0;32m Done! Train model saved at:'+save_model_path)
    plt.plot(range(len(loss_list)),loss_list)
    plt.show()
    np.save('loss_list.npy',np.array(loss_list))
    m = input("finish training!")
train()