import torch
import torchvision


import scipy as sp
from scipy.linalg import svdvals

import numpy as np
import powerlaw

# import sklearn
# from sklearn.decomposition import TruncatedSVD

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from torch import nn, Tensor
from torchvision.datasets import CIFAR10

import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn

from torch.nn.utils import prune as prune

from torch.utils.data.sampler import SubsetRandomSampler

import pickle as pickle

from torchsummary import summary

import cvxpy as cvx

import scipy as sp

from scipy import stats as stats

import seaborn as seaborn

import torchsummary

from torchsummary import summary

if __name__ == '__main__':
    device = torch.device('cuda')



    model1 = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_vgg16_bn", pretrained=True)
    model2 = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_vgg19_bn", pretrained=True)
    model3 = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet56", pretrained=True)

    # model1.to(device='cuda')
    # model1.to(device='cuda')
    # model1.to(device='cuda')


    # Load Dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


    #######################



    train_set = torchvision.datasets.CIFAR10('./datasets', train=True, 
                                            download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10('./datasets', train=False, 
                                            download=True, transform=transform)
    # Number of subprocesses to use for data loading
    num_workers = 4
    # How many samples per batch to load
    batch_size = 32
    # Percentage of training set to use as validation
    valid_size = 0.5

    num_test = len(test_set)
    indices = list(range(num_test))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_test))
    test_idx, valid_idx = indices[split:], indices[:split]

    # Define samplers for obtaining training and validation batches
    test_sampler = SubsetRandomSampler(test_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)


    # Prepare data loaders (combine dataset and sampler)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=256, 
                                            num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(test_set, batch_size= 1, sampler=valid_sampler, 
                                            num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=256, sampler=test_sampler, num_workers=num_workers)

    dataiter2 = iter(valid_loader)
    dataiter = iter(test_loader)



    # conv_dict_feats = [3,7,10,14,17,21,24,28]           #VGG11
    # conv_dict_feats = [3,7,10,14,17,21,24,28]
    conv_dict_feats = [2,6,9,13,16,19,23,26,29,33,36,39,43]

    # conv_dict = [0,4,8,11,15,18,22,25]
    conv_dict = [0,3,7,10,14,17,20,24,27,30,34,37,40]