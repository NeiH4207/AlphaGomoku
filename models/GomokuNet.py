import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from torch.distributions import Categorical
from AdasOptimizer.adasopt_pytorch import Adas
from torch.optim import Adam, SGD
from collections import deque
from tqdm import tqdm
from models.nnet import NNet
from matplotlib import pyplot as plt
from models import ModelConfig
config = ModelConfig

# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
    
# ResNet
class ResNet(nn.Module):
    def __init__(self, config, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = config['conv4-num-filter']
        self.device = 'cuda' if T.cuda.is_available() else 'cpu'
        self.layers = []
        self.layers.append(self.make_layer(block, config['conv4-num-filter'], layers[0]))
        for i in range(len(layers) - 1):
            self.layers.append(self.make_layer(block, config['conv4-num-filter']*(2**i), layers[i+1], stride=2))
        self.avg_pool = nn.AvgPool2d(2)
        
    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride).to(self.device),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        out = self.avg_pool(x)
        
        return out
    
class GomokuNet(NNet):
    def __init__(self, name, input_shape, output_shape):
        # game params
        self.name = name
        config = ModelConfig[name]
        self.config = config
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.in_channels = input_shape[0]
        self.board_x, self.board_y = input_shape[1], input_shape[2]
        self.device = 'cuda' if T.cuda.is_available() else 'cpu'

        super(GomokuNet, self).__init__()
        self.conv1 = nn.Conv2d(self.in_channels , config['conv1-num-filter'], kernel_size=config['conv1-kernel-size'], 
                               stride=config['conv1-stride'], padding=config['conv1-padding']).to(self.device)
        self.conv2 = nn.Conv2d(config['conv1-num-filter'], config['conv2-num-filter'], kernel_size=config['conv2-kernel-size'],
                                 stride=config['conv2-stride'], padding=config['conv2-padding']).to(self.device)
        self.conv3 = nn.Conv2d(config['conv2-num-filter'], config['conv3-num-filter'], kernel_size=config['conv3-kernel-size'],
                                    stride=config['conv3-stride'], padding=config['conv3-padding']).to(self.device)
        self.conv4 = nn.Conv2d(config['conv3-num-filter'], config['conv4-num-filter'], kernel_size=config['conv4-kernel-size'],
                                    stride=config['conv4-stride'], padding=config['conv4-padding']).to(self.device)
        
        self.bn1 = nn.BatchNorm2d(config['conv1-num-filter']).to(self.device)
        self.bn2 = nn.BatchNorm2d(config['conv2-num-filter']).to(self.device)
        self.bn3 = nn.BatchNorm2d(config['conv3-num-filter']).to(self.device)
        self.bn4 = nn.BatchNorm2d(config['conv4-num-filter']).to(self.device)
        
        self.resnet = ResNet(config, ResidualBlock, [2, 2, 2]).to(self.device)  
        
        self.out_conv1_dim = int((self.board_x - config['conv1-kernel-size'] + 2 * config['conv1-padding']) / config['conv1-stride'] + 1)
        self.out_conv2_dim = int((self.out_conv1_dim - config['conv2-kernel-size'] + 2 * config['conv2-padding']) / config['conv2-stride'] + 1)
        self.out_conv3_dim = int((self.out_conv2_dim - config['conv3-kernel-size'] + 2 * config['conv3-padding']) / config['conv3-stride'] + 1)
        self.out_conv4_dim = int((self.out_conv3_dim - config['conv4-kernel-size'] + 2 * config['conv4-padding']) / config['conv4-stride'] + 1)
        self.out_resnet_dim = (((self.out_conv4_dim + 1) >> 1) + 1) >> 1
        
        self.last_dim = self.out_resnet_dim * self.out_resnet_dim * config['conv4-num-filter']
        
        self.flatten_dim = self.last_dim 

        # self.last_channel_size = config.num_channels * (self.board_x - 4) * (self.board_y - 4)
        self.fc1 = nn.Linear(self.last_dim, config['fc1-num-units']).to(self.device)
        self.fc_bn1 = nn.BatchNorm1d(config['fc1-num-units']).to(self.device)

        self.fc2 = nn.Linear(config['fc1-num-units'], config['fc2-num-units']).to(self.device)
        self.fc_bn2 = nn.BatchNorm1d(config['fc2-num-units']).to(self.device)

        self.fc3 = nn.Linear(self.last_dim, config['fc3-num-units']).to(self.device)
        self.fc_bn3 = nn.BatchNorm1d(config['fc3-num-units']).to(self.device)

        self.fc4 = nn.Linear(config['fc3-num-units'], config['fc4-num-units']).to(self.device)
        self.fc_bn4 = nn.BatchNorm1d(config['fc4-num-units']).to(self.device)

        self.fc5 = nn.Linear(config['fc2-num-units'], output_shape).to(self.device)
        self.fc6 = nn.Linear(config['fc4-num-units'], 1).to(self.device)
        
        self.set_optimizer(config["optimizer"], config["learning-rate"])
        
    def forward(self, s):
        #                                                           s: batch_size x n_inputs x board_x x board_y
        s = s.view(-1, self.in_channels, self.board_x, self.board_y)    # batch_size x n_inputs x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))                          # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn2(self.conv2(s)))                          # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn3(self.conv3(s)))                          # batch_size x num_channels x (board_x-2) x (board_y-2)
        s = F.relu(self.bn4(self.conv4(s)))                          # batch_size x num_channels x (board_x-4) x (board_y-4)
        s = F.relu(self.resnet(s))
        pi = s.view(-1, self.last_dim)
        pi = F.dropout(F.relu(self.fc1(pi)), p=self.config['fc1-dropout'], training=self.training)  # batch_size x 1024
        pi = F.relu(self.fc2(pi))  # batch_size x 512
    
        v = s.view(-1, self.last_dim)
        v = F.dropout((F.relu(self.fc3(v))), p=self.config['fc3-dropout'], training=self.training)  # batch_size x 1024
        v = F.relu(self.fc4(v)) # batch_size x 512
        pi = self.fc5(pi)                                                                         # batch_size x action_size
        v = self.fc6(v)                                              # batch_size x 1

        return F.log_softmax(pi, dim=1), T.tanh(v)
    
