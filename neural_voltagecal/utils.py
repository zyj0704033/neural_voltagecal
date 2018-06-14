#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 17:34:34 2018

@author: zyj0704033
"""
import torch
import torch.utils.data as utils_data
import scipy.io
import os
import random
import numpy as np

class Mydataset(utils_data.Dataset):
    def __init__(self, data_dir='../data/train',shuffle=True,norm=True,voltage_nodes=[0,5,11,22,24,28]):
        self.__dir = data_dir
        self.__P = scipy.io.loadmat(os.path.join(self.__dir,'Pout.mat'))['Pout']
        self.__Q = scipy.io.loadmat(os.path.join(self.__dir,'Qout.mat'))['Qout']
        self.__V = (scipy.io.loadmat(os.path.join(self.__dir,'V.mat'))['V'].T)[:,voltage_nodes]
        self.__outlist = list(range(len(self.__P)))
        if shuffle:
            random.shuffle(self.__outlist)
        if norm:
            self.__V = self.normalize(self.__V)
            self.__Q = self.normalize(self.__Q)


    def __getitem__(self,index):
        Pi = self.__P[self.__outlist[index],:]
        Qi = self.__Q[self.__outlist[index],:]
        Vi = self.__V[self.__outlist[index],:]
        Pi = torch.Tensor(Pi)
        Qi = torch.Tensor(Qi)
        Vi = torch.Tensor(Vi)
        
        return [Pi,Qi,Vi]

    def __len__(self):
        return len(self.__outlist)
    
    def normalize(self,inarray):
        '''
        input type: np.ndarray ch*s
        output type: np.ndarray normalized array (in-mean)
        '''
        return (inarray-np.mean(inarray,axis=0))/np.std(inarray,axis=0)