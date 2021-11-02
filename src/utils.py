"""
@author: Vu Quoc Hien <NeiH4207@gmail.com>
"""

import numbers
import numpy as np
import os
import torch
import shutil
import torch.autograd as Variable
import matplotlib.pyplot as plt
from collections import deque
from multiprocessing import Process, Queue

class Multiprocessor():

    def __init__(self):
        self.processes = []
        self.queue = Queue()

    @staticmethod
    def _wrapper(func, queue, args, kwargs):
        ret = func(*args, **kwargs)
        queue.put(ret)

    def run(self, func, *args, **kwargs):
        args2 = [func, self.queue, args, kwargs]
        p = Process(target=self._wrapper, args=args2)
        self.processes.append(p)
        p.start()

    def wait(self):
        rets = []
        for p in self.processes:
            ret = self.queue.get()
            rets.append(ret)
        for p in self.processes:
            p.join()
        return rets
    
class dotdict(dict):
    def __getattr__(self, name):
        return self[name]

def dtanh(x):
    return 1 / np.cosh(x) ** (0.2)

# print([dtanh(i) for i in range(10)])
class AverageMeter(object):
    """From https://github.com/pytorch/examples/blob/master/imagenet/main.py"""

    def __init__(self, max_len = 100):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.vals = deque(maxlen = max_len)
        self.mean_vals = deque(maxlen = 10000)

    def __repr__(self):
        return f'{self.avg:.2e}'

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.vals.append(val)
        self.mean_vals.append(np.mean(self.vals))
    
    def plot(self, vtype = ''):
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.grid()
        ax.plot(self.mean_vals, color='red', label='trainning')
        ax.set_xlabel('Episode', fontsize=16)
        ax.set_ylabel(vtype, fontsize=16)
    
        # plt.savefig('./Experiments/' + vtype + '.pdf',bbox_inches='tight')
        plt.show()
        
# print([dtanh(i) for i in range(10)])
class AverageMeter2(object):
    """From https://github.com/pytorch/examples/blob/master/imagenet/main.py"""

    def __init__(self, max_len = 100):
        self.val1 = 0
        self.val2 = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.vals = [deque(maxlen = max_len), deque(maxlen = max_len)]
        self.mean_vals = [deque(maxlen = 10000), deque(maxlen = 10000)]

    def __repr__(self):
        return f'{self.avg:.2e}'

    def update(self, val1, val2):
        self.val1 = val1
        self.val2 = val2
        self.vals[0].append(val1)
        self.vals[1].append(val2)
        self.mean_vals[0].append(np.mean(self.vals[0]))
        self.mean_vals[1].append(np.mean(self.vals[1]))
    
    def plot(self, vtype = ''):
        plt.rcParams["figure.figsize"] = (6,3)
        plt.plot(self.mean_vals[0], color='red', label='bot 1')
        plt.plot(self.mean_vals[1], color='blue', label='bot 2')
        plt.legend()
        # plt.savefig('./Experiments/' + vtype + '.pdf',bbox_inches='tight')
        plt.show()
        


def vizualize(arr, name, cl = 'red'):
#     ax.set_yticks(np.arange(0, 1.04, 0.15))
    ax = plt.figure(num=1, figsize=(4, 3), dpi=200).gca()
    ax.set_xticks(np.arange(0, 100000, 10000))
    plt.grid()
    plt.xlabel("Iteration")
    plt.ylabel("Loss value")
#     plt.xlim(-3, 3)
#     plt.ylim(1000, 1220)
    plt.plot(arr, color = cl, linewidth = 0.9)
    # plt.legend(bbox_to_anchor=(0.785, 1), loc='upper left', borderaxespad=0.1)
    # name = name + '.pdf'
    plt.savefig(name,bbox_inches='tight')
    plt.show()
    
def plot(values, vtype = 'Scores'):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.grid()
    ax.set_xlabel('Episode')
    ax.set_ylabel(vtype)
    ax.plot(values[0], color='red', label='Bot 1')
    ax.plot(values[1], color='blue', label='Bot 2')
    ax.set_xlabel('Episode', fontsize=16)
    ax.set_ylabel(vtype, fontsize=16)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = os.path.abspath(os.path.join(dir_path, os.pardir)) + '/Experiments/' 
    if os.path.exists(dir_path):
        plt.savefig(dir_path + vtype + '.pdf',bbox_inches='tight')
    plt.show()

def plot_elo(ratings):
    
    # plotting the points
    plt.plot(ratings)
    
    # naming the x axis
    plt.xlabel('Epoch')
    # naming the y axis
    plt.ylabel('elo rating')
    
    # function to show the plot
    plt.show()


def flatten(data):
    return np.array(data).reshape(-1, )

def soft_update(target, source, tau):
	"""
	Copies the parameters from source network (x) to target network (y) using the below update
	y = TAU*x + (1 - TAU)*y
	:param target: Target network (PyTorch)
	:param source: Source network (PyTorch)
	:return:
	"""
	for target_param, param in zip(target.parameters(), source.parameters()):
		target_param.data.copy_(
			target_param.data * (1.0 - tau) + param.data * tau
		)


def hard_update(target, source):
	"""
	Copies the parameters from source network to target network
	:param target: Target network (PyTorch)
	:param source: Source network (PyTorch)
	:return:
	"""
	for target_param, param in zip(target.parameters(), source.parameters()):
			target_param.data.copy_(param.data)


def save_training_checkpoint(state, is_best, episode_count):
	"""
	Saves the models, with all training parameters intact
	:param state:
	:param is_best:
	:param filename:
	:return:
	"""
	filename = str(episode_count) + 'checkpoint.path.rar'
	torch.save(state, filename)
	if is_best:
		shutil.copyfile(filename, 'model_best.pth.tar')


# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:

	def __init__(self, action_dim, mu = 0, theta = 0.15, sigma = 0.2):
		self.action_dim = action_dim
		self.mu = mu
		self.theta = theta
		self.sigma = sigma
		self.X = np.ones(self.action_dim) * self.mu

	def reset(self):
		self.X = np.ones(self.action_dim) * self.mu

	def sample(self):
		dx = self.theta * (self.mu - self.X)
		dx = dx + self.sigma * np.random.randn(len(self.X))
		self.X = self.X + dx
		return self.X
