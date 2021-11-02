"""
Created on Fri Nov 27 16:00:47 2020
@author: hien
"""
from __future__ import division

import logging
from src.environment import Environment, Board
from src.model import Policy
from src.CaroNet import CaroNet
from src.Coach import Coach
from src.utils import dotdict
log = logging.getLogger(__name__)

args = dotdict({
    'run_mode': 'train',
    'visualize': False,
    'height':  10,
    'width': 10,
    'show_screen': False,
    'n_rows': 4,
    'exp_rate': 0.3,
    'numIters': 1000,
    'nCompare': 30,
    'numEps': 10,               # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 8,         #
    'updateThreshold': 0.51,    # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 10000,     # Number of game examples to train the neural networks.
    'numMCTSSims':10,           # Number of games moves for MCTS to simulate.
    'arenaCompare': 40,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,
    'colab_train': False,
    'colab_dir': "/content/drive/MyDrive/trainned_model/agent_mcts.pt",
    'checkpoint': './temp/',
    'load_model': True,
    'load_folder_file_1': ('Models','nnet3.pt'),
    'load_folder_file_2': ('Models','pnet3.pt'),
    'numItersForTrainExamplesHistory': 10,
    'saved_model': True
})

def main():
    env = Environment(args)
    model = CaroNet(env)
    
    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file_1)
        model.load_checkpoint(args.load_folder_file_1[0], args.load_folder_file_1[1])
    else:
        log.warning('Not loading a checkpoint!')

    # log.info('Loading the Coach...')
    coach = Coach(env, model, args)

    if args.load_model:
        # log.info("Loading 'trainExamples' from file...")
        coach.loadTrainExamples()

    # log.info('Starting the learning process !')
    for i in range(0, args.numIters):
        # bookkeeping
        print(f'Starting Iter #{i} ...')
        coach.learn(i)

if __name__ == "__main__":
    main()