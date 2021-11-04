"""
Created on Fri Nov 27 16:00:47 2020
@author: hien
"""
from __future__ import division

import logging
from src.environment import Environment
from src.model import Policy
from src.GomokuNet import GomokuNet
from src.Coach import Coach
from src.utils import dotdict
log = logging.getLogger(__name__)

args = dotdict({
    'run_mode': 'train', # train or test
    'visualize': False,
    'height':  6,
    'width': 6,
    'show_screen': True, 
    'n_in_rows': 4, 
    'exp_rate': 0.0,            # experimental rate
    '_is_selfplay': True,       # if true, then self-play, else, then test
    'numIters': 1000,           # number of iterations
    'nCompare': 50,             # Number of games to play during arena play to determine if new net will be accepted.
    'numEps': 50,               # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.5,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 10000,     # Number of game examples to train the neural networks.
    'numMCTSSims': 50,          # Number of games moves for MCTS to simulate.
    'cpuct': 3,                 # a heuristic value used to balance exploration and exploitation.
    'checkpoint': './temp/',    # Directory to save the checkpoints.
    'load_model': True,         # Whether to load the pre-trained model.
    'load_folder_file': ('Models','nnet3.pt'), # (folder,file) to load the pre-trained model from.
    'numItersForTrainExamplesHistory': 10,
    'saved_model': True         # Whether to save the model.
})

def main():
    env = Environment(args)
    nnet = GomokuNet(env)
    pnet = GomokuNet(env)
    
    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file)
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
        pnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    coach = Coach(env, nnet, pnet, args)

    if args.load_model:
        coach.loadTrainExamples()

    # log.info('Starting the learning process !')
    for i in range(0, args.numIters):
        # bookkeeping
        print(f'Starting Iter #{i} ...')
        coach.learn(i)

if __name__ == "__main__":
    main()