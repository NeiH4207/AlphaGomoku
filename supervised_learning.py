
from copy import deepcopy
from src.environment import Environment

from src.utils import dotdict
from src.CaroNet import CaroNet
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
    'height': 12,
    'width': 12,
    "n_rows": 5,
    'depth_minimax': 3,
    'show_screen': True,
    'num_iters': 1000,
    'num_epochs': 20,
    'nCompare': 30,
    'updateThreshold': 0.51,
    'mem_size': 20000,
    'mode': 'test-machine',
    'numMCTSSims': 20,          
    'arenaCompare': 40,        
    'cpuct': 1,
    'load_model': False,
    'saved_model': True,
    'load_folder_file_1': ('Models','nnet2.pt'),
    'load_folder_file_2': ('Models','pnet2.pt'),
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
            env.players[player].n_wins += 1
    return [board[1], board[0]]

def main():
    nnet = CaroNet(env)
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
        for _ in tqdm(range(args.num_epochs), desc="Self Play"):
            history = []
            board = deepcopy(env.board)
            player = choice([0, 1])
            while True:
                if player == 0:
                    probs = machine.predict(board)
                    action = action = np.argmax(probs)
                    x, y = env.convert_action_i2c(action)
                    # valids = env.get_valid_moves(board)
                    # probs = nnet.predict(board).tolist()[0]
                    # probs = probs * valids
                    # action = np.argmax(probs)
                    # x, y = env.convert_action_i2c(action)
                else:
                    probs = machine.predict(board)
                    action = action = np.argmax(probs)
                    x, y = env.convert_action_i2c(action)
                    # action = machine.predict(board)
                    # x, y = action
                    # action = env.convert_action_c2i(action)
                    # probs = [0] * env.n_actions
                    # probs[action] = 1
                    
                sym_board, sym_prob = env.get_symmetric(board, probs)
                history.append([sym_board, sym_prob, player])
                board = next_step(board, player, (x, y))
                # env.log_state(board, ('X', 'O') if player == 0 else ('O', 'X'))
                game_over, return_value = env.get_game_ended(board, action)
                if game_over:
                    if args.show_screen:
                        env.render()
                    for x in history:
                        if x[2] == player:
                            state, pi, v = x[0], x[1], 1
                        else:
                            state, pi, v = x[0], x[1], -1
                            valids = env.get_valid_moves(state)
                            n_valids = np.sum(valids)
                            valids[action] = 0
                            if return_value == 0 or n_valids == 0: continue
                            pi += valids / n_valids
                        if return_value == 0:
                            v = 0
                        trainExamples.append([state, pi, v])    
                        
                    if args.show_screen:
                        env.render()
                        env.screen.reset()
                        env.render()
                    break
                player = 1 - player
                if args.show_screen:
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
        pnet = CaroNet(env)
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
            nnet.save_checkpoint(folder=args.load_folder_file_1[0], 
                                 filename='rejected_' + args.load_folder_file_1[1])
            nnet = pnet
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