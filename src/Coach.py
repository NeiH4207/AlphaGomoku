import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from models.GomokuNet import GomokuNet
from src.machine import Machine
from random import shuffle
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

    def __init__(self, game, players, maxlenOfQueue=1000, numEps=10, tempThreshold=1,
                 show_screen=False,numItersForTrainExamplesHistory=15, updateThreshold=0.8, cpuct=1,
                 numMCTSSims=15, exploration_rate=0.25, checkpoint=None, train_epochs=20,
                 batch_size=32, loss_func='bce', n_compares=30, speed=0.2, load_folder_file=None,):
        
        self.game = game
        self.players = players
        self.maxlenOfQueue = maxlenOfQueue
        self.numEps = numEps
        self.tempThreshold = tempThreshold
        self.cpuct = cpuct  
        self.numMCTSSims = numMCTSSims
        self.exploration_rate = exploration_rate
        self.show_screen = show_screen
        self.numItersForTrainExamplesHistory = numItersForTrainExamplesHistory
        self.checkpoint = checkpoint
        self.updateThreshold = updateThreshold
        self.train_epochs = train_epochs
        self.batch_size = batch_size
        self.loss_func = loss_func
        self.n_compares = n_compares
        self.speed = speed
        self.load_folder_file = load_folder_file
        
        self.trainExamplesHistory = []  # history of examples from numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()
        self.scores = AverageMeter2()
        
        for player in players:
            player.set_loss_function(loss_func)
        
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
            temp = int(episodeStep < self.tempThreshold)
            pi = self.mcts.getActionProb(board, temp)
            sym_boards, sym_pis = self.game.get_symmetric(board, pi)
            for sym_board, sym_pi in zip(sym_boards, sym_pis):
                trainExamples.append([sym_board.get_state(), sym_pi, player])
            
            action = np.random.choice(len(pi), p=pi)
            board = self.game.get_next_state(board, action, player, render=self.show_screen)
            terminate, r = self.game.get_game_ended(board, action, episodeStep)
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
        print('Current NNET ELO:', self.players[0].get_elo())
        self.mcts = MCTS(self.game, self.players[0],
                         numMCTSSims=self.numMCTSSims,
                         cpuct=self.cpuct,
                         exploration_rate=self.exploration_rate,
                         selfplay=True)
        self.game.reset()
        # examples of the iteration
        if not self.skipFirstSelfPlay or iter > 0:
            self.iterationTrainExamples = deque([], maxlen=self.maxlenOfQueue)

            for _ in tqdm(range(self.numEps), desc="Self Play"):
                self.executeEpisode()

            # save the iteration examples to the history 
            self.trainExamplesHistory.append(self.iterationTrainExamples)

        while len(self.trainExamplesHistory) > self.numItersForTrainExamplesHistory:
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
        
        shuffle(trainExamples)
        before_elo = self.players[0].get_elo()
        # training new network, keeping a copy of the old one
        self.players[0].learn(trainExamples, epochs=self.train_epochs, batch_size=self.batch_size)
        eval = Evaluation(self.game, players=self.players, n_compares=self.n_compares,
                          show_screen=self.show_screen, speed=self.speed)

        print('PITTING AGAINST PREVIOUS VERSION')
        nwins, pwins, draws = eval.run()

        print('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
        print('NNET ELO:', self.players[0].get_elo())
        print('PNET ELO:', self.players[1].get_elo())
        if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.updateThreshold:
            print('REJECTING NEW MODEL')
            self.players[0].save_model(folder=self.load_folder_file[0], 
                                 filename='rejected_' + self.load_folder_file[1])
            self.players[0].set_model(GomokuNet(name=self.players[0].nnet.name, input_shape=self.game.nnet_input_shape, output_shape=self.game.n_actions))
            self.players[0].load_model(folder=self.load_folder_file[0], 
                                 filename= self.load_folder_file[1])
            self.trainExamplesHistory.pop(-1)
        else:
            print('ACCEPTING NEW MODEL')
            self.players[0].save_model(folder=self.load_folder_file[0], 
                                      filename=self.load_folder_file[1])
            
            self.players[1].load_model(folder=self.load_folder_file[0], 
                                      filename=self.load_folder_file[1])
            self.players[1].set_elo(self.players[0].get_elo())            

    def getCheckpointFile(self):
        return 'checkpoint_' + 'pt'

    def saveTrainExamples(self):
        folder = self.checkpoint
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
