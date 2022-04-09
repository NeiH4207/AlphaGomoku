
from copy import deepcopy
from src.environment import Environment

from src.utils import dotdict
from models.GomokuNet import GomokuNet
from src.machine import Machine
from src.evaluate import Evaluation
from src.MCTS import MCTS
from collections import deque
from random import shuffle, seed
from torchsummary import summary
from tqdm import tqdm
import logging
import numpy as np
from random import choice
log = logging.getLogger(__name__)

args = dotdict({
    'height': 6,
    'width': 6,
    "n_in_rows": 4,
    'depth_minimax': 3,
    'show_screen': True,
    'exploration_rate': 0.04,    # exploration rate for self-play
    '_is_selfplay': True,       # if true, then self-play, else, then test
    'num_iters': 1000,
    'num_epochs': 30,
    'nCompare': 100,
    'updateThreshold': 0.4,
    'mem_size': 20000,
    'mode': 'test-machine',
    'numMCTSSims': 160,          
    'arenaCompare': 40,        
    'cpuct': 1,
    'load_model': False,
    'saved_model': True,
    'load_folder_file_1': ('Models','nnet6x6.pt'),
    'load_folder_file_2': ('Models','pnet6x6.pt'),
    'algo': 'mcts'
})

env = Environment(args)

def next_step(board, player, action):
    x, y = action
    if board[0][x][y] + board[1][x][y] == 0:
        if env.show_screen:
            env.screen.draw(x, y, player)
        board[0][x][y] = 1
        if env.check_game_ended(board, 0, action):
            env.players[player].score += 1
    return [board[1], board[0]]

def main():
    nnet = GomokuNet(env)
    # summary(nnet, (args.height, args.width))
    
    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file_1[0], args.load_folder_file_1[1])
    
    if args.algo == 'mcts':
        machine = MCTS(env, nnet, args)
    else:
        machine = Machine(env, nnet)
    replay_memories = deque([], maxlen=args.mem_size)
    print('NNET ELO:', nnet.elo)
    
    for iter in range(args.num_iters):
        # -------
        trainExamples = deque([], maxlen=args.mem_size)
        machine.reset()
        for _ in tqdm(range(args.num_epochs), desc="Self Play"):
            history = []
            board = env.get_new_board()
            player = choice([0, 1])
            env.restart()   
            while True:
                if player == 0:
                    probs = machine.predict(board)
                    # action = np.argmax(probs)
                    # probs = machine.predict(board)
                    action = np.random.choice(len(probs), p=probs)
                    probs = [0] * env.n_actions
                    probs[action] = 1
                    # probs = nnet.predict(board.get_state())
                    # action = np.argmax(probs)
                    # probs = [0] * env.n_actions
                    # probs[action] = 1
            
                    act = env.convert_action_v2i(action)
                    sym_boards, sym_pis = env.get_symmetric(board, probs)
                    for sym_board, sym_pi in zip(sym_boards, sym_pis):
                        history.append([sym_board, sym_pi, action, player])
                else:
                    probs = machine.predict(board)
                    
                    # probs = machine.predict(board)
                    action = np.random.choice(len(probs), p=probs)
                    probs = [0] * env.n_actions
                    probs[action] = 1
                    # probs = nnet.predict(board)
                    # action = np.argmax(probs)
                    # probs = [0] * env.n_actions
                    # probs[action] = 1
                    sym_boards, sym_pis = env.get_symmetric(board, probs)
                    for sym_board, sym_pi in zip(sym_boards, sym_pis):
                        history.append([sym_board, sym_pi, action, player])
                    
                board = env.get_next_state(board, action, player, render=args.show_screen)
                # env.log_state(board, ('X', 'O') if player == 0 else ('O', 'X'))
                game_over, return_value = env.get_game_ended(board, action)
                if game_over:
                    env.render()
                    for x in history:
                        if x[3] == player:
                            _board, pi, act, v = x[0], x[1], x[2], 1
                        else:
                            _board, pi, act, v = x[0], x[1], x[2], -1
                        if return_value == 0:
                            v = 0
                        trainExamples.append([_board.get_state(), pi, v]) 
                    break
                player = 1 - player
                env.render()
        
        # shuffle examples before training
        replay_memories.append(trainExamples)
        
        # training new network, keeping a copy of the old one
        train_data = []
        for e in replay_memories:
            train_data.extend(e)
        print('NUM OBS CLAMED:' ,len(train_data))
        if len(train_data) < nnet.args.batch_size: continue
        
        shuffle(train_data)
        nnet.save_checkpoint(folder=args.load_folder_file_1[0], 
                             filename=args.load_folder_file_1[1])
        pnet = GomokuNet(env)
        pnet.load_checkpoint(args.load_folder_file_1[0], args.load_folder_file_1[1])
        
        # training new network, keeping a copy of the old one
        nnet.train_examples(train_data)
        env.render()
        eval = Evaluation(env, nnet, pnet)
        eval.run()

        print('PITTING AGAINST PREVIOUS VERSION')
        nwins, pwins, draws = eval.get_info()

        print('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
        if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < args.updateThreshold:
            print('REJECTING NEW MODEL')
            print('NNET ELO:', nnet.elo)
            print('PNET ELO:', pnet.elo)
            nnet.save_checkpoint(folder=args.load_folder_file_1[0], 
                                 filename='rejected_' + args.load_folder_file_1[1])
            nnet.load_checkpoint(args.load_folder_file_1[0], args.load_folder_file_1[1])
        else:
            print('ACCEPTING NEW MODEL')
            print('NNET ELO:', nnet.elo)
            print('PNET ELO:', pnet.elo)
            nnet.save_checkpoint(folder=args.load_folder_file_1[0], 
                                 filename=args.load_folder_file_1[1])
            pnet.save_checkpoint(folder=args.load_folder_file_2[0], 
                                 filename=args.load_folder_file_2[1])
        
    
if __name__ == "__main__":
    main()