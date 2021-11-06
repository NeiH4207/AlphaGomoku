import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from src.machine import Machine
from random import shuffle
from src.machine import Machine
from src.evaluate import Evaluation
import numpy as np
from tqdm import tqdm
from src.utils import AverageMeter2
from src.MCTS import MCTS

log = logging.getLogger(__name__)


class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, nnet, pnet, args):
        self.game = game
        self.nnet = nnet
        self.pnet = pnet  # the competitor network
        self.args = args
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()
        self.scores = AverageMeter2()
        
    def executeEpisode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard, currPlayer, pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamples = []
        self.game.restart()
        board = self.game.get_new_board()
        episodeStep = 0
        player = 0
        
        while True:
            episodeStep += 1
            temp = int(episodeStep < self.args.tempThreshold)
            pi = self.mcts.getActionProb(board, temp)
            sym_boards, sym_pis = self.game.get_symmetric(board, pi)
            for sym_board, sym_pi in zip(sym_boards, sym_pis):
                trainExamples.append([sym_board.get_state(), sym_pi, player])
            
            action = np.random.choice(len(pi), p=pi)
            board = self.game.get_next_state(board, action, player, render=self.args.show_screen)
            terminate, r = self.game.get_game_ended(board, action)
            if terminate:
                if r != 0:
                    self.game.players[player].score += 1
                
                self.iterationTrainExamples += [(x[0], x[1], r * ((-1) ** (x[2] == player))) 
                                                for x in trainExamples]
                break
            
            player = 1 - player

    def learn(self, iter):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """
        print('Current NNET ELO:', self.nnet.elo)
        self.mcts = MCTS(self.game, self.nnet, self.args)  # reset search tree
        self.game.reset()
        # examples of the iteration
        if not self.skipFirstSelfPlay or iter > 0:
            self.iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

            for _ in tqdm(range(self.args.numEps), desc="Self Play"):
                self.executeEpisode()
                
            if self.args.visualize:
                self.scores.plot()

            # save the iteration examples to the history 
            self.trainExamplesHistory.append(self.iterationTrainExamples)

        while len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
            log.warning(
                f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
            self.trainExamplesHistory.pop(0)
            
        # backup history to a file
        self.saveTrainExamples()

        # shuffle examples before training
        trainExamples = []
        for e in self.trainExamplesHistory:
            trainExamples.extend(e)
        print('NUM OBS CLAMED:' ,len(trainExamples))
        if len(trainExamples) < self.nnet.args.batch_size: return
        
        shuffle(trainExamples)
        
        # training new network, keeping a copy of the old one
        self.nnet.train_examples(trainExamples)
        eval = Evaluation(self.game, self.nnet, self.pnet)

        print('PITTING AGAINST PREVIOUS VERSION')
        nwins, pwins, draws = eval.run()

        print('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
        print('NNET ELO:', self.nnet.elo)
        print('PNET ELO:', self.pnet.elo)
        if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.args.updateThreshold:
            print('REJECTING NEW MODEL')
            self.nnet.save_checkpoint(folder=self.args.load_folder_file[0], 
                                 filename='rejected_' + self.args.load_folder_file[1])
            self.nnet.load_checkpoint(folder=self.args.load_folder_file[0], 
                                 filename= self.args.load_folder_file[1])
            self.trainExamplesHistory.pop(-1)
        else:
            print('ACCEPTING NEW MODEL')
            self.nnet.save_checkpoint(folder=self.args.load_folder_file[0], 
                                      filename=self.args.load_folder_file[1])
            
            self.pnet.load_checkpoint(folder=self.args.load_folder_file[0], 
                                      filename=self.args.load_folder_file[1])
    def getCheckpointFile(self):
        return 'checkpoint_' + 'pt'

    def saveTrainExamples(self):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile() + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed
        print('\nTrain Examples saved succesful!')

    def loadTrainExamples(self):
        folder = self.args.checkpoint
        modelFile = os.path.join(folder, self.getCheckpointFile())
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            log.info("File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            log.info('Loading done!')

            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True
