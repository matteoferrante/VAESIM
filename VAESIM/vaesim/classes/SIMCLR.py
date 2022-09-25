import numpy as np
import shutil, time, os, requests, random, copy

import torch
import torch.nn as nn
import torch.optim as optim

from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
from torchvision import datasets, transforms, models
from torchvision.transforms import *

import matplotlib.pyplot as plt

import tqdm

from sklearn.manifold import TSNE

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from yellowbrick.cluster import KElbowVisualizer

from sklearn.metrics.cluster import normalized_mutual_info_score
from utils.utils import linear_assignment

from scipy import stats
from sklearn.metrics import classification_report,accuracy_score
from sklearn.metrics import confusion_matrix

import seaborn as sns
from torch.nn.functional import softmax
from IPython.display import clear_output
import wandb
from utils.aesimclr_evaluation import *

import medmnist
from medmnist import INFO, Evaluator
from monai import transforms as m_transforms

import pandas as pd




class LinearLayer(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 use_bias = True,
                 use_bn = False,
                 **kwargs):
        super(LinearLayer, self).__init__(**kwargs)

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.use_bn = use_bn
        
        self.linear = nn.Linear(self.in_features, 
                                self.out_features, 
                                bias = self.use_bias and not self.use_bn)
        if self.use_bn:
             self.bn = nn.BatchNorm1d(self.out_features)

    def forward(self,x):
        x = self.linear(x)
        if self.use_bn:
            x = self.bn(x)
        return x




class ProjectionHead(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features,
                 out_features,
                 head_type = 'nonlinear',
                 **kwargs):
        super(ProjectionHead,self).__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.head_type = head_type

        if self.head_type == 'linear':
            self.layers = LinearLayer(self.in_features,self.out_features,False, True)
        elif self.head_type == 'nonlinear':
            self.layers = nn.Sequential(
                LinearLayer(self.in_features,self.hidden_features,True, True),
                nn.ReLU(),
                LinearLayer(self.hidden_features,self.out_features,False,True))
        
    def forward(self,x):
        x = self.layers(x)
        return x
    
    
    
class SimCLR(nn.Module):
    def __init__(self,base_model="resnet-18",in_features=2048,
                 hidden_features=2048,
                 out_features=128,
                 head_type = 'nonlinear',n_channels=1):
        super().__init__()
        self.base_model = base_model
        
        #PRETRAINED MODEL
        self.pretrained = models.resnet50(pretrained=True)
        
        
        
        self.pretrained.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), bias=False)
        self.pretrained.maxpool = nn.Identity()
        
        self.pretrained.fc = nn.Identity()
        
        for p in self.pretrained.parameters():
            p.requires_grad = False
        
        
        if n_channels==1:
            self.pretrained.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), bias=False)
            self.pretrained.conv1.requires_grad=True
            
        
        self.projector = ProjectionHead(in_features, hidden_features, out_features,head_type)

    def forward(self,x):
        h = self.pretrained(x)
        
        z = self.projector(torch.squeeze(h))
        
        return h,z
