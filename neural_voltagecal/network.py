#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 17:44:34 2018

@author: zyj0704033
"""

import torch.nn as nn

class FCnet(nn.Module):
    def __init__(self):
        super(FCnet,self).__init__()
        self.fc1 = nn.Linear(32,100)
        self.fc2 = nn.Linear(100,100)
        self.fc3 = nn.Linear(100,100)
        self.fc4 = nn.Linear(100,6)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
    def forward(self,x):
        y = self.tanh(self.fc1(x))
        y = self.tanh(self.fc2(y))
        y = self.tanh(self.fc3(y))
        y = self.fc4(y)
        
        return y


class Convnet(nn.Module):
    def __init__(self):
        super(Convnet,self).__init__()
        self.conv1 = nn.Conv1d(1,16,9,padding=4)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu1 = nn.ReLU()
        #16*32
        self.conv2 = nn.Conv1d(16,32,7,padding=3)
        self.bn2 = nn.BatchNorm1d(32)
        self.relu2 = nn.ReLU()
        #32*32
        self.conv3 = nn.Conv1d(32,32,7,padding=3)
        self.bn3 = nn.BatchNorm1d(32)
        self.relu3 = nn.ReLU()
        #32*32
        self.fc1 = nn.Linear(1024,100)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(100,1)
    def forward(self,x):
        y = self.bn1(self.conv1(x))
        y = self.relu1(y)
        y = self.bn2(self.conv2(y))
        y = self.relu2(y)
        y = self.bn3(self.conv3(y))
        y = self.relu3(y)
        y = y.view(-1,1024)
        y = self.fc1(y)
        y = self.relu4(y)
        y = self.fc2(y)

        return y


class FCConvnet(nn.Module):
    def __init__(self):
        super(FCConvnet,self).__init__()
        self.fc0 = nn.Linear(32,225)
        self.conv1 = nn.Conv2d(1,16,5,padding=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16,32,5,padding=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32,32,5,padding=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(7200,500)
        self.fc2 = nn.Linear(500,6)
        self.relu = nn.ReLU()
    def forward(self,x):
        y = self.relu(self.fc0(x))
        y = y.view(-1,1,15,15)
        y = self.conv1(y)
        y = self.relu(self.bn1(y))
        y = self.conv2(y)
        y = self.relu(self.bn2(y))
        y = self.conv3(y)
        y = self.relu(self.bn3(y))
        y = y.view(-1,7200)
        y = self.relu(self.fc1(y))
        y = self.fc2(y)

        return y

