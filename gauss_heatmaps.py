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



def get_data(num_samples, in_module, wt_module, dataiter2):
    
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
    
    

    with torch.no_grad():


        ss = wt_module.weight.shape[0]
        tensor_list = [torch.empty(ss, 0).to(device) for _ in range(10)]
        for j in range(num_samples):

            
            
            try:
                image,label = next(dataiter2)
            except StopIteration:
                dataiter2 = iter(valid_loader)
                image,label = next(dataiter2)

            #
            activation = {}
            def getActivation(name):
              # the hook signature
                def hook(model, input, output):
                    activation[name] = output.detach()
                return hook
            

            h1 = in_module.register_forward_hook(getActivation('feats'))


            output = model(image.to(device))
            Xx1 = activation['feats']

            h1.remove()
            r,c,ch,f = Xx1.shape
            print
            class_means = torch.zeros(c)
            
            X1 = torch.sum(Xx1,(0,2,3)).to(device) / (ch*f)
#             X1 = torch.flatten(Xx1, start_dim=2)
#             X2 = torch.sum(X1, (0,2))
            tensor_list[label] = torch.cat((tensor_list[label], X1.unsqueeze(1)), dim=1)
    
#     p_list =np.zeros((ss,10))
#     stat_list = np.zeros((ss,10))
#     for i in range(10):
#         temp1 = tlist[i]
#         temp1.cpu()

#         temp1.detach()
#         temp1.cpu().numpy()
#         for j in range(s):
#             T = temp1.cpu().numpy()[j,:]
#     #         print(T.shape)
#             stat, p = stats.kstest(T, 'norm')
#     #         stat, p = stats.shapiro(T)
#     #         print(stat,p)
#             p_list[j,i] = p
#             stat_list[j,i] = stat
    
    
    
    
    return ss, tensor_list
            
            
            
            
            
if __name__ == '__main__':
    device = torch.device('cuda')



    model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_vgg16_bn", pretrained=True)
    model.to(device='cuda')


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

    for rr in range(len(conv_dict)):
        print('layer '+str(rr))
        in_module = model.features[conv_dict_feats[rr]]
        wt_module = model.features[conv_dict[rr]]

        s, tlist = get_data(512, in_module, wt_module, dataiter2)

    
#     temp1 = tlist[i][60,:]
#     temp1.cpu()
    
#     temp1.detach()
#     temp1.cpu().numpy()
#     counts, bins = np.histogram(temp1.cpu().detach().numpy(), bins=40)
#     print(tlist[i][60,:].shape, np.mean(temp1.cpu().detach().numpy()), np.var(temp1.cpu().detach().numpy()))
#     plt.stairs(counts,bins)
#     plt.show()
    
        p_list =np.zeros((s,10))
        stat_list = np.zeros((s,10))
        for i in range(10):
            temp1 = tlist[i]
            temp1.cpu()

            temp1.detach()
            temp1.cpu().numpy()
            for j in range(s):
                T = temp1.cpu().numpy()[j,:]
        #         print(T.shape)
        #         stat, p = stats.kstest(T, stats.norm.cdf, alternative='less')
                stat, p = stats.shapiro(T)
        #         print(stat,p)
                p_list[j,i] = p
                stat_list[j,i] = stat
                
    #     for i in range(10):
    #         h = np.arange(s)
        
            
    # #         plt.show()
    #         bad = sum(k > .1 for k in p_list[:,i]) / s
    #         print(bad)
    #         print([k for k, v in enumerate(p_list[:,i]) if v > 0.1])
    #         plt.title('Layer ' + str(rr) + ' class ' + str(i) + ' p values with Gaussian fraction = '+str(bad))
    #         plt.plot(h, p_list[:,i])
    #         filename1 = 'pvals_vgg_16_trained_layer_'+str(rr)+'_class_'+str(i)+'.png'
    #         plt.savefig(filename1)
    #         plt.clf()
        
        
        idx = np.random.choice(s, 15, replace=False)
        
        seaborn.heatmap(p_list[idx,:], vmin = 0, vmax = 1, cmap='viridis')
        plt.title('Layer '+str(rr)+' Gaussianity')
        # seaborn.heatmap(np.abs(vgg16_trained[i]/np.linalg.norm(vgg16_trained[i])), vmin=vmin_[i]*0, vmax= vmax_[i], cmap='crest')
        
        # plt.show()
        filename2 = 'vgg_16_trained_layer_'+str(+rr)+'_15ilts_gaussheatmap3.png'
        plt.savefig(filename2)
        plt.clf()