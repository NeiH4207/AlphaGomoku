
from src.environment import Environment

from src.utils import dotdict, plot_elo
from src.model import Policy
from src.CaroNet import CaroNet
from src.machine import Machine
import time
import sys
import pygame
import numpy as np

args = dotdict({
    'height': 10,
    'width': 10,
    "n_rows": 5,
    'show_screen': True,
    'mode': 'test-machine',
    'model': 'nnet',
    'load_folder_file': ('Models','nnet3.pt')
})

env = Environment(args)

def next_step(board, player, action):
    x, y = action
    if board[0][x][y] + board[1][x][y] == 0:
        env.screen.draw(x, y, player)
        board[0][x][y] = 1
        env.screen.render()
        if env.check_game_ended(board, 0, action):
            env.players[player].n_wins += 1
            env.screen.reset()
    return [board[1], board[0]]

def main():
    if args.mode == 'test-machine':
        if args.model == 'nnet':
            machine = CaroNet(env)
            machine.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
            plot_elo(machine._elo)
        elif args.model == 'ai-engine':
            machine = Machine(env)
            
        game_over = False
        # -------
        player = 1
        board = env.board
        while True:
            if player == 1:
                probs = machine.predict(board)
                valids = env.get_valid_moves(board)
                probs = probs * valids
                action = env.convert_action_i2c(np.argmax(probs))
                x, y = action
                board = next_step(board, player, action)
                player = 1 - player
            else:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        sys.exit()
                    if event.type == pygame.MOUSEBUTTONDOWN and not game_over:
                        mouseX = event.pos[1] # x
                        mouseY = event.pos[0] # y
                        
                        x = int(mouseY // env.screen.SQUARE_SIZE)
                        y = int(mouseX // env.screen.SQUARE_SIZE)
                        if board[0][x][y] + board[1][x][y] == 0:
                            board = next_step(board, player, (x, y))
                            player = 1 - player
    else:   
        env.play()
    
if __name__ == "__main__":
    main()