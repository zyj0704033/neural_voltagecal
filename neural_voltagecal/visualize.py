#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 10:22:22 2018

@author: zyj0704033
"""

import scipy.io
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from network import Convnet,FCnet
from torch.autograd import Variable

voltage_nodes=[0,5,11,22,24,28]
loss_list = np.load('loss_list.npy')
plt.plot(loss_list)
plt.show()
#m = input()

Q = scipy.io.loadmat('../data/train/Qoutrs.mat')['Qoutrs']
Q_mean = np.mean(Q,axis=0)
Q_std = np.std(Q,axis=0)
Qn = (Q-Q_mean)/Q_std


V = scipy.io.loadmat('../data/train/Vrs.mat')['Vrs'].T
V_mean = np.mean(V,axis=0)
V_std = np.std(V,axis=0)
Vn = (V-V_mean)/V_std

model = Convnet()
model.load_state_dict(torch.load('../data/model/Convnet_epoch_200_Tue_Jun_19_15:28:28_2018.model'))
predict = np.zeros((len(Qn),6))
for i in range(len(Qn)):
    testin = Qn[i,:]
    testin = torch.Tensor(testin)
    testin = testin.view(1,1,-1)
    testin = Variable(testin)
    testout = model.forward(testin)
    predict[i,:] = testout.data[0].numpy()

result = predict*V_std[voltage_nodes]+V_mean[voltage_nodes]
plt.figure(1)
plt.plot(V[:,1])
plt.plot(result[:,1])