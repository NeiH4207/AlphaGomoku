

import logging
from random import choice, seed
from src.environment import Environment

from src.elo_helper import compute_elo
from src.model import Policy
from tqdm import tqdm
import numpy as np
log = logging.getLogger(__name__)

class Evaluation():
    
    def __init__(self, game, nnet = None, pnet = None) -> None:
        self.game = game
        self.nnet = nnet
        self.pnet = pnet
        self.n_wins = [0, 0]
        self.n_draws = 0
        self.n_battles = 0
        
        game.players[0].reset()
        game.players[1].reset()
        
    def load_model(self, nnet, pnet):
        self.nnet = nnet
        self.pnet = pnet
        
    def get_info(self):
        return self.n_wins[0], self.n_wins[1], self.n_draws
    
    def get_elo(self):
        return self.nnet.elo, self.pnet.elo
    
    def play(self):
        for iter in tqdm(range(self.game.args.nCompare), desc="Evaluating"):
            game_over = False
            player = choice([0, 1])
            board = self.game.board
            while not game_over:
                valids = self.game.get_valid_moves(board)
                if player == 0:
                    probs = self.nnet.predict(board.get_state())[0]
                    probs = probs * valids
                    action = self.game.convert_action_i2c(np.argmax(probs))
                else:
                    probs = self.pnet.predict(board.get_state())[0]
                    probs = probs * valids
                    action = self.game.convert_action_i2c(np.argmax(probs))
                    
                board = self.game.next_state(board, action, player, 
                                             render=self.game.args.show_screen)
                # self.game.log_state(board, ('X', 'O') if player == 0 else ('O', 'X'))
                action = self.game.convert_action_c2i(action)
                game_over, result = self.game.get_game_ended(board, action)
                if game_over:
                    self.n_battles += 1
                    self.n_wins[player] += 1
                    w = 0.5
                    if player == 0: 
                        w = 1
                    else: 
                        w = 0
                    if result == 0: 
                        self.n_draws += 1
                        w = 0.5
                    r0, r1 = compute_elo(self.nnet.elo, self.pnet.elo, w)
                    self.nnet.update_elo(r0)
                    self.pnet.update_elo(r1)
                player = 1 - player
            
            if self.game.show_screen:
                self.game.screen.reset()
                
        self.game.players[0].reset()
        self.game.players[1].reset()
        
    # evaluate the model
    def run(self, nnet, pnet):
        """
        nnet: neural network
        pnet: competitor neural network
        """
        self.load_model(nnet, pnet)
        self.play()
        return self.get_info()