import logging
from random import choice

from src.elo_helper import compute_elo
from tqdm import tqdm
import numpy as np
log = logging.getLogger(__name__)

class Evaluation():
    
    def __init__(self, game, nnet, pnet):
        self.nnet = nnet
        self.pnet = pnet
        self.game = game
        self.game.reset()
        self.score = [0, 0]
        self.n_draws = 0
        self.n_battles = 0
        
    def load_model(self, nnet, pnet):
        self.nnet = nnet
        self.pnet = pnet
        self.score = [0, 0]
        self.n_draws = 0
        self.n_battles = 0
        
    def get_info(self):
        return self.score[0], self.score[1], self.n_draws
    
    def get_elo(self):
        return self.nnet.elo, self.pnet.elo
    
    def play(self):
        for _ in tqdm(range(self.game.args.nCompare), desc="Evaluating"):
            game_over = False
            player = choice([0, 1])
            board = self.game.get_new_board()
            self.game.restart()
            while not game_over:
                valids = self.game.get_valid_moves(board)
                if player == 0:
                    probs = self.nnet.predict(board.get_state())[0]
                    probs = probs * valids
                    action = np.argmax(probs)
                else:
                    probs = self.pnet.predict(board.get_state())[0]
                    probs = probs * valids
                    action = np.argmax(probs)
                    
                board = self.game.get_next_state(
                    board=board, 
                    action=action, 
                    playerID=player, 
                    render=self.game.args.show_screen)
                # self.game.log_state(board, ('X', 'O') if player == 0 else ('O', 'X'))
                game_over, result = self.game.get_game_ended(board, action)
                if game_over:
                    self.n_battles += 1
                    if result != 0:
                        self.score[player] += 1
                        self.game.players[player].score += 1
                        if player == 0:
                            w = 1
                        else:
                            w = 0
                    else:
                        w = 0.5
                    # recompute elo
                    r0, r1 = compute_elo(self.nnet.elo, self.pnet.elo, w)
                    self.nnet.update_elo(r0)
                    self.pnet.update_elo(r1)
                self.game.render
                player = 1 - player
                
    
        
    # evaluate the model
    def run(self):
        """
        nnet: neural network
        pnet: competitor neural network
        """
        self.play()
        return self.get_info()