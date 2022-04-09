"""
Created on Fri Nov 27 16:00:47 2020
@author: hien
"""
from __future__ import division
import argparse

import logging
from src.Coach import Coach
from src.environment import Environment
from src.player import Player
log = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_mode', type=str, default='train', 
                        help='train or test')
    parser.add_argument('--visualize', type=bool, default=False, 
                        help='visualize the game')
    parser.add_argument('--height', type=int, default=5, 
                        help='height of the board')
    parser.add_argument('--width', type=int, default=5, 
                        help='width of the board')
    parser.add_argument('--show_screen', type=bool, default=True, 
                        help='show the screen')
    parser.add_argument('--speed', type=float, default=0, 
                        help='speed of the game')
    parser.add_argument('--n_in_rows', type=int, default=5, 
                        help='number of consecutive stones in a row to win')
    parser.add_argument('--exploration_rate', type=float, default=0.1, 
                        help='exploration rate for self-play')
    parser.add_argument('--exp_rate', type=float, default=0.2, 
                        help='experimental rate')
    parser.add_argument('--_is_selfplay', type=bool, default=True,
                        help='if true, then self-play, else, then test')
    parser.add_argument('--numIters', type=int, default=1000,
                        help='number of iterations')
    parser.add_argument('--nCompare', type=int, default=50, 
                        help='Number of games to play during arena play to determine if new net will be accepted.')
    parser.add_argument('--numEps', type=int, default=20,
                        help='Number of complete self-play games to simulate during a new iteration.')
    parser.add_argument('--tempThreshold', type=int, default=10, 
                        help='tempThreshold')
    parser.add_argument('--updateThreshold', type=float, default=0.5,
                        help='During arena playoff, new neural net will be accepted if threshold or more of games are won.')
    parser.add_argument('--maxlenOfQueue', type=int, default=5000,
                        help='Number of game examples to train the neural networks.')
    parser.add_argument('--numMCTSSims', type=int, default=50, 
                        help='Number of games moves for MCTS to simulate.')
    parser.add_argument('--cpuct', type=float, default=1, 
                        help='a heuristic value used to balance exploration and exploitation.')
    parser.add_argument('--checkpoint', type=str, default='./temp/', 
                        help='Directory to save the checkpoints.')
    parser.add_argument('--trainEpochs', type=int, default=10,
                        help='Number of epochs to train the neural network.')
    parser.add_argument('--trainBatchSize', type=int, default=32,
                        help='Batch size for training.')
    parser.add_argument('--load_model', type=bool, default=True, 
                        help='Whether to load the pre-trained model.')
    parser.add_argument('--load_folder_file', type=list, default=['trainned_models','nnet'], 
                        help='(folder,file) to load the pre-trained model from.')
    parser.add_argument('--numItersForTrainExamplesHistory', type=int, default=10,
                        help='Number of iterations to store the trainExamples history.')
    parser.add_argument('--saved_model', type=bool, default=True, 
                        help='Whether to save the model.')
    args = parser.parse_args()
    args.load_folder_file[1] = args.load_folder_file[1] + str(args.height) + 'x' + str(args.width) + '.pt'
    return args

def main():
    args = parse_args()
    env = Environment(args.height, args.width, args.show_screen,
                      n_in_rows=args.n_in_rows)
    players = [Player(name=str(i)) for i in range(2)]
    env.set_players(players)
    
    coach = Coach ( game=env, 
                    players=players, 
                    numEps=args.numEps, 
                    tempThreshold=args.tempThreshold,
                    updateThreshold=args.updateThreshold,
                    maxlenOfQueue=args.maxlenOfQueue,
                    numMCTSSims=args.numMCTSSims,
                    exploration_rate=args.exploration_rate,
                    cpuct=args.cpuct,
                    show_screen=args.show_screen,
                    numItersForTrainExamplesHistory=args.numItersForTrainExamplesHistory,
                    checkpoint=args.checkpoint,
                    train_epochs=args.trainEpochs,
                    batch_size=args.trainBatchSize,
                    n_compares=args.nCompare,
                    speed=args.speed,
                    load_folder_file=args.load_folder_file,
                )
    
    for i in range(0, args.numIters):
        # bookkeeping
        print(f'Starting Iter #{i} ...')
        coach.learn(i)
        
if __name__ == "__main__":
    main()