import logging
from random import choice

from src.elo_helper import compute_elo
from tqdm import tqdm
import numpy as np
log = logging.getLogger(__name__)
import time

class Evaluation():
    
    def __init__(self, game, players=None, n_compares=100, speed=0, show_screen=False):
        self.players = players
        self.game = game
        self.n_compares = n_compares
        self.game.reset()
        self.score = [0, 0]
        self.n_draws = 0
        self.n_battles = 0
        self.show_screen = show_screen
        self.speed = speed
        
    def load_model(self, nnet, pnet):
        self.nnet = nnet
        self.pnet = pnet
        self.score = [0, 0]
        self.n_draws = 0
        self.n_battles = 0
        
    def get_info(self):
        return self.score[0], self.score[1], self.n_draws
    
    def get_elo(self):
        return self.players[0].get_elo(), self.players[1].get_elo()
    
    def play(self):
        for _ in tqdm(range(self.n_compares), desc="Evaluating"):
            game_over = False
            playerID = choice([0, 1])
            board = self.game.get_new_board()
            self.game.restart()
            while not game_over:
                valids = self.game.get_valid_moves(board)
                if playerID == 0:
                    action = self.players[0].get_action(board.get_state(), validMoves=valids)
                else:
                    action = self.players[1].get_action(board.get_state(), validMoves=valids)
                    
                board = self.game.get_next_state(
                    board=board, 
                    action=action, 
                    playerID=playerID, 
                    render=self.show_screen)
                # self.game.log_state(board, ('X', 'O') if player == 0 else ('O', 'X'))
                game_over, result = self.game.get_game_ended(board, action)
                if game_over:
                    self.n_battles += 1
                    if result != 0:
                        self.score[playerID] += 1
                        self.game.players[playerID].score += 1
                        if playerID == 0:
                            w = 1
                        else:
                            w = 0
                    else:
                        w = 0.5
                    # recompute elo
                    r0, r1 = compute_elo(self.players[0].get_elo(), self.players[1].get_elo(), w)
                    self.players[0].set_elo(r0)
                    self.players[1].set_elo(r1)
                self.game.render()
                playerID = 1 - playerID
                time.sleep(self.speed)
                
    
        
    # evaluate the model
    def run(self):
        """
        nnet: neural network
        pnet: competitor neural network
        """
        self.play()
        return self.get_info()