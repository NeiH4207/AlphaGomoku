import os
from tqdm import tqdm
import numpy as np
import torch as T
from torch.autograd import Variable
import torch.nn.functional as F
from src.utils import AverageMeter


# Gomoku player class
class Player:
    def __init__(self, name = None, symbol = None, color = None):
        self.name = name
        self.symbol = symbol
        self.color = color
        self.score = 0
        self.last_move = None
        self.n_moves = 0
        self.nnet = None
        self.elo = 0
        self.pi_losses = AverageMeter()
        self.v_losses = AverageMeter()
    
    def set_elo(self, elo):
        self.elo = elo
    
    def get_elo(self):
        return self.elo
    
    def set_model(self, model):
        self.nnet = model
        
    def get_model(self):
        return self.nnet
    
    def reset_score(self):
        self.score = 0

    def set_score(self, score):
        self.score = score
        
    def predict(self, state):
        return self.nnet.predict(state)
    
    def get_action(self, board=None, validMoves=None, getBestMove=False):
        probs, _ = self.nnet.predict(board)
        probs = probs * validMoves
        sum_probs = np.sum(probs)
        probs = probs / sum_probs
        if getBestMove:
            return np.argmax(probs)
        else:
            return np.random.choice(range(len(probs)), p=probs)
    
    
    def learn(self, examples=None, lr=0.001, epochs=20, batch_size=32):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        for epoch in range(epochs):
            # print('\nEPOCH ::: ' + str(epoch + 1))
            self.nnet.train()
            batch_count = int(len(examples) / batch_size)
            t = tqdm(range(batch_count), desc='Training Net')
            self.nnet.set_learning_rate(lr)
            for _ in t:
                sample_ids = np.random.randint(len(examples), size=batch_size)
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                boards = np.array(boards, dtype = np.float32)\
                    .reshape(-1, boards[0].shape[0], boards[0].shape[1], boards[0].shape[2])
                boards = Variable(T.FloatTensor(boards.astype(np.float64)).to(self.nnet.device))
                target_pis = Variable(T.FloatTensor(np.array(pis).astype(np.float64))).to(self.nnet.device)
                target_vs = Variable(T.FloatTensor(np.array(vs).astype(np.float64))).to(self.nnet.device)

                # compute output
                out_pi, out_v = self.nnet(boards)
                l_pi = self.nnet.loss_pi(target_pis, out_pi)
                l_v = self.nnet.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                # record loss
                self.pi_losses.update(l_pi.item(), boards.size(0))
                self.v_losses.update(l_v.item(), boards.size(0))
                # compute gradient and do Adas step
                self.nnet.reset_grad()
                total_loss.backward()
                self.nnet.optimize()
                entropy = -T.mean(
                    T.sum(T.exp(out_pi) * out_pi, 1)
                ).item()
                t.set_postfix(loss_pi=self.pi_losses, loss_v=self.v_losses, etp=entropy)
    
    def save_model(self, folder=None, filename=None):
        # create folder if it doesn't exist
        if not os.path.exists(folder):
            os.makedirs(folder)
            
        path = folder + '/' + filename
        self.nnet.save(path)
        
    def load_model(self, folder=None, filename=None):
        path = folder + '/' + filename
        self.nnet.load(path)