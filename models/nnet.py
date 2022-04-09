import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from torch.distributions import Categorical
from AdasOptimizer.adasopt_pytorch import Adas
from torch.optim import Adam, SGD
from collections import deque
from tqdm import tqdm
from src.utils import dotdict, AverageMeter, plot
from torch.autograd import Variable
from matplotlib import pyplot as plt
from models import ModelConfig
from torch import optim as optim

class NNet(nn.Module):

    def forward(self, x):
        pass
            
    def predict(self, x):
        x = torch.FloatTensor(np.array(x.cpu())).to(self.device).detach()		# Chuyển đầu ra x về dạng torch tensor
        x = x.reshape(-1, self.input_size)
        output = self.forward(x)
        return output.cpu().data.numpy().flatten()
 
    def set_loss_function(self, loss):
        if loss == "mse":
            self.loss = nn.MSELoss()		# Hàm loss là tổng bình phương sai lệch
        elif loss == "cross_entropy":
            self.loss = nn.CrossEntropyLoss()
        elif loss == "bce":
            self.loss = nn.BCELoss()		# Hàm loss là binary cross entropy, với đầu ra 2 lớp
        elif loss == "bce_logits":
            self.loss = nn.BCEWithLogitsLoss()		# Hàm BCE logit sau đầu ra dự báo có thêm sigmoid, giống BCE
        elif loss == "l1":
            self.loss = nn.L1Loss()
        elif loss == "smooth_l1":
            self.loss = nn.SmoothL1Loss()		# Hàm L1 loss nhưng có đỉnh được làm trơn, khả vi với mọi điểm
        elif loss == "soft_margin":
            self.loss = nn.SoftMarginLoss()		# Hàm tối ưu logistic loss 2 lớp của mục tiêu và đầu ra dự báo
        else:
            raise ValueError("Loss function not found")
        
    def set_optimizer(self, optimizer, lr):
        if optimizer == "sgd":
            self.optimizer = optim.SGD(self.parameters(), lr=lr)		# Tối ưu theo gradient descent thuần túy
        elif optimizer == "adam":
            self.optimizer = optim.Adam(self.parameters(), lr=lr)
        elif optimizer == "adadelta":
            self.optimizer = optim.Adadelta(self.parameters(), lr=lr)		# Phương pháp Adadelta có lr update
        elif optimizer == "adagrad":
            self.optimizer = optim.Adagrad(self.parameters(), lr=lr)		# Phương pháp Adagrad chỉ cập nhật lr ko nhớ
        elif optimizer == "rmsprop":
            self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
            
    def predict(self, s):
        s = torch.FloatTensor(np.array(s)).to(self.device).detach()
        with torch.no_grad():
            self.eval()
            pi, v = self.forward(s)
            return torch.exp(pi).cpu().numpy()[0], v.item()
    
    def set_learning_rate(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr    
            
    def reset_grad(self):
        self.optimizer.zero_grad()
        
    def step(self):
        self.optimizer.step()
        
    def optimize(self):
        self.optimizer.step()
        
    def loss_pi(self, targets, outputs):
        return -torch.mean(torch.sum(targets * outputs, 1))

    def loss_v(self, targets, outputs):
        return F.mse_loss(outputs.view(-1), targets)
          
    def save_train_losses(self, train_losses):
        self.train_losses = train_losses
        
    def save(self, path=None):
        torch.save(self.state_dict(), path)
        print("Model saved at {}".format(path))
        
    def load(self, path=None):
        if path is None:
            raise ValueError("Path is not defined")
        self.load_state_dict(torch.load(path, map_location=torch.device(self.device)))
        print('Model loaded from {}'.format(path))
