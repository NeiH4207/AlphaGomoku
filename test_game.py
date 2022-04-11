
import argparse
from random import random
from src.player import Player
from src.environment import Environment
from src.machine import Machine
from collections import deque
import time
import sys
import pygame
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str, default='nnet3x3', 
                        help='name of the model')
    parser.add_argument('--model', type=str, default='nnet', 
                        help='nnet or ai-engine')
    parser.add_argument('--mode', type=str, default='test-model',
                        help='test-model or test-selfplay')
    parser.add_argument('--height', type=int, default=3, 
                        help='height of the board')
    parser.add_argument('--width', type=int, default=3, 
                        help='width of the board')
    parser.add_argument('--show_screen', type=bool, default=True, 
                        help='show the screen')
    parser.add_argument('--speed', type=float, default=0, 
                        help='speed of the game')
    parser.add_argument('--n_in_rows', type=int, default=3, 
                        help='number of consecutive stones in a row to win')
    parser.add_argument('--_is_selfplay', type=bool, default=True,
                        help='if true, then self-play, else, then test')
    parser.add_argument('--numIters', type=int, default=1000,
                        help='number of iterations')
    parser.add_argument('--nCompare', type=int, default=50, 
                        help='Number of games to play during arena play to determine if new net will be accepted.')
    parser.add_argument('--load_model', type=bool, default=True, 
                        help='Whether to load the pre-trained model.')
    parser.add_argument('--train', action='store_true',
                        help='realtime training')
    parser.add_argument('--load_folder_file', type=list, default=['trainned_models','nnet'], 
                        help='(folder,file) to load the pre-trained model from.')
    args = parser.parse_args()
    args.model_name = args.load_folder_file[1] + str(args.height) + 'x' + str(args.width)
    args.load_folder_file[1] = args.model_name + '.pt'
    return args

def main():
    args = parse_args()
    env = Environment(args.height, args.width, args.show_screen,
                      n_in_rows=args.n_in_rows)
    players = [Player(name=str(i)) for i in range(2)]
    env.set_players(players, model_name=args.model_name)
    machine = players[0]
    if args.mode == 'test-model':
        if args.model == 'nnet':
            machine.load_model(folder=args.load_folder_file[0], 
                                      filename=args.load_folder_file[1])
            # plot_elo(machine._elo)
        elif args.model == 'ai-engine':
            machine = Machine(env)
            
        game_over = False
        player = np.random.choice([0, 1])
        board = env.get_new_board()
        trainExamples = deque([], maxlen=200)
        history = []
        while True:
            # Get action from player
            x, y = None, None
            if player == 1:
                valids = env.get_valid_moves(board)
                action = machine.get_action(board.get_state(), validMoves=valids, getBestMove=True)
                action = env.convert_action_i2xy(action)
                x, y = action
            else:
                events = pygame.event.get() 
                for event in events:
                    if event.type == pygame.QUIT:
                        if args.train:
                            players[0].save_model(folder=args.load_folder_file[0], 
                                            filename=args.load_folder_file[1])
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
            
            if args.train:  
                probs = np.zeros(board.get_state()[0].shape)
                probs[x][y] = 1
                sym_boards, sym_pis = env.get_symmetric(board, probs)
                for sym_board, sym_pi in zip(sym_boards, sym_pis):
                    history.append([sym_board, sym_pi, action, player])
                
            board = env.get_next_state(board, action, player, render=args.show_screen)
            game_over, result = env.get_game_ended(board, env.convert_action_c2i(action))
            if game_over:
                if result == 1:
                    env.players[1 - player].score += 1
                elif result == -1:
                    env.players[player].score += 1
                    
                for x in history:
                    if x[3] == player:
                        _board, pi, act, v = x[0], x[1], x[2], 1
                    else:
                        _board, pi, act, v = x[0], x[1], x[2], -1
                    if args.train:
                        trainExamples.append([_board.get_state(), pi, v]) 
                
                if args.train:
                    players[0].learn(trainExamples, epochs=1, batch_size=len(trainExamples))
                
                    history = []
                board = env.get_new_board()
                time.sleep(1)
                env.restart()
                env.render()
                player = np.random.choice([0, 1])
            else:
                player = 1 - player
    else:   
        env.play()
    
if __name__ == "__main__":
    main()