
from random import random
from src.environment import Environment
from src.utils import dotdict, plot_elo
from src.model import Policy
from models.SimpleNet import GomokuNet
from src.machine import Machine
import time
import sys
import pygame
import numpy as np

args = dotdict({
    'height': 5,
    'width': 5,
    "n_in_rows": 4,
    'show_screen': True,
    'mode': 'test-model',
    'model': 'nnet',
    'load_folder_file': ('Models','nnet5x5.pt')
})

def main():
    # Initialize environment
    env = Environment(args)
    if args.mode == 'test-model':
        if args.model == 'nnet':
            machine = GomokuNet(env)
            machine.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
            # plot_elo(machine._elo)
        elif args.model == 'ai-engine':
            machine = Machine(env)
            
        game_over = False
        player = np.random.choice([0, 1])
        board = env.get_new_board()
        while True:
            # Get action from player
            x, y = None, None
            if player == 1:
                probs = machine.predict(board.get_state())
                valids = env.get_valid_moves(board)
                probs = probs * valids
                action = env.convert_action_i2xy(np.argmax(probs))
                x, y = action
            else:
                events = pygame.event.get() 
                for event in events:
                    if event.type == pygame.QUIT:
                        sys.exit()
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        mouseX = event.pos[1] # x
                        mouseY = event.pos[0] # y
                        x = int(mouseY // env.screen.SQUARE_SIZE)
                        y = int(mouseX // env.screen.SQUARE_SIZE)
                        pygame.event.clear()
               
            if x is not None and y is not None and env.is_valid_move(board, x, y):
                action = (x, y) 
            else:
                continue
            x, y = action
            board = env.get_next_state(board, action, player, render=args.show_screen)
            game_over, result = env.get_game_ended(board, env.convert_action_c2i(action))
            if game_over:
                if result == 1:
                    env.players[1 - player].score += 1
                elif result == -1:
                    env.players[player].score += 1
                time.sleep(1)
                board = env.get_new_board()
                env.restart()
                env.render()
                player = np.random.choice([0, 1])
            else:
                player = 1 - player
    else:   
        env.play()
    
if __name__ == "__main__":
    main()