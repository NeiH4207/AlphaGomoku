
import argparse
from src.environment import Environment
from src.player import Player

from models.GomokuNet import GomokuNet
from src.machine import Machine
from src.evaluate import Evaluation
from src.MCTS import MCTS
from collections import deque
from tqdm import tqdm
import logging
import numpy as np
from random import choice
log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str, default='nnet3x3', 
                        help='name of the model')
    parser.add_argument('--visualize', type=bool, default=False, 
                        help='visualize the game')
    parser.add_argument('--height', type=int, default=13, 
                        help='height of the board')
    parser.add_argument('--width', type=int, default=13, 
                        help='width of the board')
    parser.add_argument('--show_screen', action='store_true', 
                        help='show the screen')
    parser.add_argument('--speed', type=float, default=0, 
                        help='speed of the game')
    parser.add_argument('--n_in_rows', type=int, default=5, 
                        help='number of consecutive stones in a row to win')
    parser.add_argument('--exploration_rate', type=float, default=0.1, 
                        help='exploration rate for self-play')
    parser.add_argument('--exp_rate', type=float, default=0.2, 
                        help='experimental rate')
    parser.add_argument('--_is_selfplay', type=bool,
                        help='if true, then self-play, else, then test')
    parser.add_argument('--numIters', type=int, default=10,
                        help='number of iterations')
    parser.add_argument('--nCompare', type=int, default=50, 
                        help='Number of games to play during arena play to determine if new net will be accepted.')
    parser.add_argument('--numEps', type=int, default=5,
                        help='Number of complete self-play games to simulate during a new iteration.')
    parser.add_argument('--tempThreshold', type=int, default=10, 
                        help='tempThreshold')
    parser.add_argument('--updateThreshold', type=float, default=0.5,
                        help='During arena playoff, new neural net will be accepted if threshold or more of games are won.')
    parser.add_argument('--maxlenOfQueue', type=int, default=5000,
                        help='Number of game examples to train the neural networks.')
    parser.add_argument('--numMCTSSims', type=int, default=500, 
                        help='Number of games moves for MCTS to simulate.')
    parser.add_argument('--cpuct', type=float, default=2.5, 
                        help='a heuristic value used to balance exploration and exploitation.')
    parser.add_argument('--checkpoint', type=str, default='./temp/', 
                        help='Directory to save the checkpoints.')
    parser.add_argument('--trainEpochs', type=int, default=2,
                        help='Number of epochs to train the neural network.')
    parser.add_argument('--trainBatchSize', type=int, default=32,
                        help='Batch size for training.')
    parser.add_argument('--loss_func', type=str, default='mse',
                        help='Loss function for training.')
    parser.add_argument('--load_model', action='store_true',
                        help='Load a saved model.')
    parser.add_argument('--load_folder_file', type=list, default=['trainned_models','nnet'], 
                        help='(folder,file) to load the pre-trained model from.')
    parser.add_argument('--numItersForTrainExamplesHistory', type=int, default=10,
                        help='Number of iterations to store the trainExamples history.')
    parser.add_argument('--saved_model', action='store_true', default=True,  
                        help='Whether to save the model.')
    parser.add_argument('--algo', type=str, default='greedy',
                        help='Which algorithm to use.')
    parser.add_argument('--mem_size', type=int, default=20000,
                        help='Size of the memory.')
    
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
    
    for player in players:
        player.set_loss_function(args.loss_func)
        if args.load_model:
            player.load_model(args.load_folder_file[0], args.load_folder_file[1])
            
    if args.algo == 'mcts':
        machine = MCTS(env, players[0],
                         numMCTSSims=args.numMCTSSims,
                         cpuct=args.cpuct,
                         exploration_rate=args.exploration_rate,
                         selfplay=False)
    else:
        machine = Machine(env, players[0])
        
    replay_memories = deque([], maxlen=args.mem_size)
    print('NNET ELO:', players[0].get_elo())
    
    for iter in range(args.numEps):
        # -------
        trainExamples = deque([], maxlen=args.mem_size)
        machine.reset()
        for _ in tqdm(range(args.numEps), desc="Self Play"):
            history = []
            board = env.get_new_board()
            player = choice([0, 1])
            env.restart()   
            while True:
                if player == 0:
                    probs = machine.predict(board)
                    action = np.random.choice(len(probs), p=probs)
                    probs = [0] * env.n_actions
                    probs[action] = 1
                    sym_boards, sym_pis = env.get_symmetric(board, probs)
                    for sym_board, sym_pi in zip(sym_boards, sym_pis):
                        history.append([sym_board, sym_pi, action, player])
                else:
                    probs = machine.predict(board)
                    action = np.random.choice(len(probs), p=probs)
                    probs = [0] * env.n_actions
                    probs[action] = 1
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
        players[0].learn(train_data, epochs=args.trainEpochs, batch_size=args.trainBatchSize)
        eval = Evaluation(env, players=players, n_compares=args.nCompare,
                          show_screen=args.show_screen, speed=args.speed)

        print('PITTING AGAINST PREVIOUS VERSION')
        nwins, pwins, draws = eval.run()

        print('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
        print('NNET ELO:', players[0].get_elo())
        print('PNET ELO:', players[1].get_elo())
        if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < args.updateThreshold:
            print('REJECTING NEW MODEL')
            players[0].save_model(folder=args.load_folder_file[0], 
                                 filename='rejected_' + args.load_folder_file[1])
            players[0].set_model(GomokuNet(name=players[0].nnet.name, input_shape=env.nnet_input_shape, output_shape=env.n_actions))
            players[0].load_model(folder=args.load_folder_file[0], 
                                 filename= args.load_folder_file[1])
        else:
            print('ACCEPTING NEW MODEL')
            players[0].save_model(folder=args.load_folder_file[0], 
                                      filename=args.load_folder_file[1])
            
            players[1].load_model(folder=args.load_folder_file[0], 
                                      filename=args.load_folder_file[1])
            players[1].set_elo(players[0].get_elo())        
        
    
if __name__ == "__main__":
    main()