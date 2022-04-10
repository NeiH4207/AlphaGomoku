import logging
import math
import numpy as np
from src.simulate import *
from scipy.special import softmax
EPS = 1e-8
log = logging.getLogger(__name__)


class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game=None, player=None, numMCTSSims=15, selfplay=True, 
                 exploration_rate=0.25, cpuct=2.5):
        self.game = game
        self.player = player
        self.numMCTSSims = numMCTSSims
        self.selfplay = selfplay
        self.exploration_rate = exploration_rate
        self.cpuct_base = 19652
        self.cpuct_init = cpuct
        
        
        self.Qsa  = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa  = {}  # stores #times edge s,a was visited
        self.Ns   = {}  # stores #times board s was visited
        self.Ps   = {}  # stores initial policy (returned by neural net)

        self.Es   = {}  # stores game.get_game_ended ended for board s
        self.Vs   = {}  # stores game.getValidMoves for board s

    def reset(self):
        self.Qsa = {}
        self.Nsa = {}
        self.Ns = {}
        self.Ps = {}
        self.Es = {}
        self.Vs = {}
        
    def get_cpuct_value(self, s):
        cpuct = math.log((self.Ns[s] + self.cpuct_base + 1) / self.cpuct_base) + self.cpuct_init
        return cpuct

    def predict(self, board, temp=1):
        return self.getActionProb(board, temp)
        
    def getActionProb(self, board, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        board.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        s = board.string_representation()
        for _ in range(self.numMCTSSims):
            self.search(board)

        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 
                  for a in range(self.game.n_actions)]
        # print([(a, self.Qsa[(s, a)]) for a in range(self.game.n_actions) if (s, a) in self.Qsa])
        # print([(a, self.Nsa[(s, a)]) for a in range(self.game.n_actions) if (s, a) in self.Nsa])
        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
        else:
            # probs = softmax(1.0/temp * np.log(np.array(counts) + 1e-10))
            counts = [x ** (1. / temp) for x in counts]
            counts_sum = float(sum(counts))
            if counts_sum == 0:
                probs = [1 / self.game.n_actions for _ in range(self.game.n_actions)]
            else:
                probs = [x / counts_sum for x in counts]
        if self.selfplay:
                # add Dirichlet Noise for exploration (needed for
                # self-play training)
                valids = self.game.get_valid_moves(board)
                dirictlet_rd = valids * np.random.dirichlet(0.3 * np.ones(len(probs)))
                # renomalize dirictlet_rd to sum to 1
                dirictlet_rd = dirictlet_rd / np.sum(dirictlet_rd)
                # add dirictlet noise to probs
                probs = np.array(probs) * (1 - self.exploration_rate) + dirictlet_rd * self.exploration_rate
        return probs
    
    def predict(self, board):
        s = board.string_representation()
        for _ in range(self.numMCTSSims):
            self.search(board)

        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 
                  for a in range(self.game.n_actions)]
        # get probabilities of actions by counts
        sum_counts = float(sum(counts))
        probs = [x / sum_counts for x in counts]
        return probs
        

    def search(self, board, last_action = None, depth = 0):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current board
        """
        s = board.string_representation()
         
        terminate = False
        if last_action != None:
            terminate, self.Es[s] = self.game.get_game_ended(board, last_action)
            if terminate: return -self.Es[s]

        if s not in self.Ps:
            # leaf node
            self.Ps[s], v = self.player.predict(board.get_state())
            # if np.random.uniform() < self.args.exp_rate:
            #     # explore
            #     probs = get_probs(board.get_state())
            #     for p in probs:
            #         a = self.game.convert_action_c2i(p)
            #         self.Ps[s][a] = probs[p]
            valids = self.game.get_valid_moves(board)
            self.Ps[s] = self.Ps[s] * valids  # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[s])
            self.Vs[s] = valids
            self.Ns[s] = 0
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s  # renormalize
                return -v
            else:
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your player architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your player and/or training process.   
                log.error("All valid moves were masked, doing a workaround.")
                log.error("Board:")
                log.error(board.to_string())
                self.Ps[s] += 1 / self.game.n_actions
                return 0
     
        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(self.game.n_actions):
            if valids[a]:
                cpuct = self.get_cpuct_value(s)
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (
                            1 + self.Nsa[(s, a)])
                else:
                    u = cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?
                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        next_s = self.game.get_next_state(board, a)
        v = self.search(next_s, a, depth + 1)
        
        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) \
                / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v
