#!/usr/bin/game python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 09:45:45 2020

@author: hien
"""
from operator import is_
import numpy as np
import torch
from src.model import Policy
from random import random, randint, choices, uniform
from src.utils import flatten
import torch.nn.functional as F
from sklearn.utils import shuffle
from copy import deepcopy as dcopy
from torch.distributions import Categorical
from src.simulate import stupid_score, best_move
import logging
import math
log = logging.getLogger(__name__)
torch.manual_seed(1)
MAP_SIZE = 5

class Machine():
    def __init__(self, game, nnet):
        self.game = game
        self.nnet = nnet
            
    def convert_one_hot(self, action):
        n_values = self.action_dim
        return np.eye(n_values, dtype = np.float32)[action]
    
    def convert_one_hot_tensor(self, next_pred_action):
        return torch.zeros(len(next_pred_action), 
                           self.action_dim).scatter_(1, next_pred_action.unsqueeze(1), 1.)
    
    def predict(self, board, mode = 'ai-engine'):
        assert mode in ['minimax','ai-engine']
        if mode == 'ai-engine':
            action = best_move(board, 0)
            # print(action)
            return action
        return False
    
    def get_stupid_score(self,board):
        value = self.nnet.get_value(board)
        return value
    
    def minimax(self, board, depth, alpha = -math.inf, beta = math.inf, maximizingPlayer = True, last_action = None):
        # self.game.log_state(board)
        # print("Depth - ", depth)
        valids = self.game.get_valid_moves(board)
        
        if last_action != None:
            is_terminate, value = self.game.get_game_ended(board, last_action)
            if is_terminate:
                return (None, value * 10000000000000)
            
            if depth == 0:
                return (None, stupid_score(board))

        valid_actions = []
        
        for i in range(self.game.n_actions):
            if valids[i] == 1:
                valid_actions.append(i)
        
        if len(valid_actions) == 0:
            return (None, 0)
        
        if maximizingPlayer:
            value = -math.inf
            action = np.random.choice(valid_actions)
            for col in valid_actions:
                next_board = self.game.get_next_state(board, col)
                
                new_score = self.minimax(next_board, depth-1, alpha, beta, False, col)[1]

                if new_score > value:
                    value = new_score
                    action = col

                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return action, value
        
        else: # Minimizing player
            value = math.inf
            action = np.random.choice(valid_actions)
            for col in valid_actions:
                next_board = self.game.get_next_state(board, col)
                new_score = self.minimax(next_board, depth-1, alpha, beta, True, col)[1]

                if new_score < value:
                    value = new_score
                    action = col

                beta = min(beta, value)
                if alpha >= beta:
                    break
            return action, value

    def get_move(self, board):
        col, minimax_score = self.minimax(board, self.depth, -math.inf, math.inf, True)
        return col
    
    def select_action_by_exp(self, state, agent, action):
        state = torch.FloatTensor(state).to(self.device)
        agent = torch.FloatTensor(agent).to(self.device)
        _, prob, state_value = self.model(state, agent)
        prob = Categorical(prob)
        log_p = prob.log_prob(torch.tensor(action).to(self.device))
        self.model.entropies += prob.entropy().mean()
        return action, log_p, state_value
                      
        
    def select_action_smart(self, state, agent_pos, game):
        score_matrix, agents_matrix, conquer_matrix, \
                       treasures_matrix, walls_matrix = [dcopy(_) for _ in state]
        actions = [0] * game.n_agents
        state = dcopy(state)
        agent_pos = dcopy(agent_pos)
        init_score = 0
        order = shuffle(range(game.n_agents))
        exp_rewards = [0] * game.n_agents
        
        for i in range(game.n_agents):
            agent_id = order[i]
            
            act = 0
            scores = [0] * 8
            mn = 1000
            mx = -1000
            valid_states = []
            for act in range(game.n_actions):
                _state, _agent_pos = dcopy([state, agent_pos])
                valid, next_state, reward = game.soft_step(agent_id, _state, act, _agent_pos, exp=True)
                scores[act] = reward - init_score
                mn = min(mn, reward - init_score)
                mx = max(mx, reward - init_score)
                valid_states.append(valid)
            
            # _scores = dcopy(scores)
            # scores[0] = mn
            # for j in range(len(scores)):
            #     scores[j] = (scores[j] - mn) / (mx - mn + 0.0001)
    
            # sum = np.sum(scores) + 0.0001
            # for j in range(len(scores)):
            #     scores[j] = scores[j] / sum
            #     if valid_states[j] is False:
            #         scores[j] = 0
            act = np.array(scores).argmax()
            valid, state, score = game.soft_step(agent_id, state, act, agent_pos, exp=True)
            init_score = score
            actions[agent_id] = act
            exp_rewards[agent_id] = mx
        return actions, exp_rewards
    
    def select_action_test_not_predict(self, state):
        actions = []
        state = dcopy(state)
        state = np.reshape(flatten(state), (7, 20, 20))
        state = [state[0], [state[1], state[2]], [state[3], state[4]], state[5], state[6]]
        agent_pos_1 = dcopy(self.game.agent_pos_1)
        agent_pos_2 = dcopy(self.game.agent_pos_2)
        init_score = self.game.score_mine - self.game.score_opponent
        rewards = []
        states = []
        next_states = []
        
        for i in range(self.num_agents):
            _state = state
            _state[1] = self.game.get_agent_state(_state[1], i)
            _state = flatten(_state)
            states.append(state)
            act = 0
            scores = [0] * 8
            mn = 1000
            mx = -1000
            valid_states = []
            for act in range(8):
                _state, _agent_pos_1, _agent_pos_2 = dcopy([state, agent_pos_1, agent_pos_2])
                valid, _state, _agent_pos, _score = self.game.fit_action(i, _state, act, _agent_pos_1, _agent_pos_2, False)
                scores[act] = _score - init_score
                mn = min(mn, _score - init_score)
                mx = max(mx, _score - init_score)
                valid_states.append(valid)
            for j in range(len(scores)):
                scores[j] = (scores[j] - mn) / (mx - mn + 0.0001)
                scores[j] **= 5
            sum = np.sum(scores) + 0.0001
            for j in range(len(scores)):
                scores[j] = scores[j] / sum
                if valid_states[j] is False:
                    scores[j] = 0
            act = choices(range(self.game.n_actions), scores)[0]
            valid, state, agent_pos, score = self.game.fit_action(i, state, act, agent_pos_1, agent_pos_2)
            init_score = score
            actions.append(act)
            next_states.append(state)
            
        return states, actions, rewards, next_states
    
    def select_random(self, state):
        actions = []
        for i in range(self.num_agents):
            actions.append(randint(0, 7))
        return state, actions, [0] * self.num_agents, state 
    
    def save_models(self):
        """
        saves the target actor and critic models
        :param episode_count: the count of episodes iterated
        :return:        
        """
        self.model.save_checkpoint()
        
    def load_models(self):
        """
        loads the target actor and critic models, and copies them onto actor and critic models
        :param episode: the count of episodes iterated (used to find the file name)
        :return:
        """
        self.model.load_checkpoint(self.args.load_folder_file[0],
                                   self.args.load_folder_file[1])
        
        self.model.eval()