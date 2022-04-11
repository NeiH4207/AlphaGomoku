#!/usr/bin/env python
import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import matplotlib
from models import ModelConfig

from models.nnet import NNet
matplotlib.use("Agg")

class ResBlock(nn.Module):
    def __init__(self, inplanes=128, planes=128, kernel_size=3, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.device = 'cuda' if T.cuda.is_available() else 'cpu'
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride,
                     padding=1, bias=False).to(self.device)
        self.bn1 = nn.BatchNorm2d(planes).to(self.device)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=stride,
                     padding=1, bias=False).to(self.device)
        self.bn2 = nn.BatchNorm2d(planes).to(self.device)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out
    
class OutBlock(nn.Module):
    def __init__(self, config, input_shape, output_shape):
        super(OutBlock, self).__init__()
        self.config = config
        self.device = 'cuda' if T.cuda.is_available() else 'cpu'
        self.fc1 = nn.Linear(input_shape, config['fc1-num-units']).to(self.device)
        self.fc_bn1 = nn.BatchNorm1d(config['fc1-num-units']).to(self.device)

        self.fc2 = nn.Linear(config['fc1-num-units'], config['fc2-num-units']).to(self.device)
        self.fc_bn2 = nn.BatchNorm1d(config['fc2-num-units']).to(self.device)

        self.fc3 = nn.Linear(config['fc2-num-units'], config['fc3-num-units']).to(self.device)
        self.fc_bn3 = nn.BatchNorm1d(config['fc3-num-units']).to(self.device)

        self.fc4 = nn.Linear(config['fc2-num-units'], output_shape).to(self.device)
        self.fc5 = nn.Linear(config['fc3-num-units'], 1).to(self.device)
    
    def forward(self, s):
        out = F.dropout(F.relu(self.fc1(s)), p=self.config['fc1-dropout'], training=self.training)  # batch_size x 1024
        out = F.relu(self.fc2(out))
        pi =self.fc4(out) 
    
        v = F.dropout((F.relu(self.fc3(out))), p=self.config['fc3-dropout'], training=self.training)  # batch_size x 1024                               # batch_size x action_size
        v = self.fc5(v)                                              # batch_size x 1

        return F.log_softmax(pi, dim=1), T.tanh(v)
    
class AZNet(NNet):
    def __init__(self, name, input_shape, output_shape):
        super(AZNet, self).__init__()
        # game params
        self.name = name
        config = ModelConfig[name]
        self.config = config
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.in_channels = input_shape[0]
        self.board_x, self.board_y = input_shape[1], input_shape[2]
        self.device = 'cuda' if T.cuda.is_available() else 'cpu'
        self.conv1 = nn.Conv2d(self.in_channels , config['conv1-num-filter'], kernel_size=config['conv1-kernel-size'], 
                               stride=config['conv1-stride'], padding=config['conv1-padding']).to(self.device)
        
        self.conv2 = nn.Conv2d(config['conv1-num-filter'], config['conv2-num-filter'], kernel_size=config['conv2-kernel-size'],
                                 stride=config['conv2-stride'], padding=config['conv2-padding']).to(self.device)
        self.conv3 = nn.Conv2d(config['conv2-num-filter'], config['conv3-num-filter'], kernel_size=config['conv3-kernel-size'],
                                    stride=config['conv3-stride'], padding=config['conv3-padding']).to(self.device)
        
        self.bn1 = nn.BatchNorm2d(config['conv1-num-filter']).to(self.device)
        self.bn2 = nn.BatchNorm2d(config['conv2-num-filter']).to(self.device)
        self.bn3 = nn.BatchNorm2d(config['conv3-num-filter']).to(self.device)
        
        self.avg_pool = nn.AvgPool2d(2)
        
        for block in range(config['num-resblocks']):
            setattr(self, "res_%i" % block,ResBlock(config['conv3-num-filter'],
                                                    config['conv3-num-filter'],
                                                    config['resblock-kernel-size']))
        
        
        self.out_conv1_dim = int((self.board_x - config['conv1-kernel-size'] + 2 * config['conv1-padding']) / config['conv1-stride'] + 1)
        self.out_conv2_dim = int((self.out_conv1_dim - config['conv2-kernel-size'] + 2 * config['conv2-padding']) / config['conv2-stride'] + 1)
        self.out_conv3_dim = int((self.out_conv2_dim - config['conv3-kernel-size'] + 2 * config['conv3-padding']) / config['conv3-stride'] + 1)
        self.flatten_dim = config['conv3-num-filter'] * ((self.out_conv3_dim >> 1) ** 2)
        self.outblock = OutBlock(config, self.flatten_dim, output_shape)
        self.set_optimizer(config["optimizer"], config["learning-rate"])
    
    def forward(self,s):
        s = s.view(-1, self.in_channels, self.board_x, self.board_y)  
        s = F.relu(self.bn1(self.conv1(s)))                          # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn2(self.conv2(s)))           # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn3(self.conv3(s))) 
        for block in range(self.config['num-resblocks']):
            s = getattr(self, "res_%i" % block)(s)
        s = self.avg_pool(s)               
        s = s.view(-1, self.flatten_dim)
        s = self.outblock(s)
        return s