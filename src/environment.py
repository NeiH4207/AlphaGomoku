from copy import deepcopy as dcopy
from numpy import array
from GameBoard.game_board import Screen
from src.utils import flatten
import numpy as np

# Gomoku player class
class Player:
    def __init__(self, name = None, symbol = None, color = None):
        self.name = name
        self.symbol = symbol
        self.color = color
        self.score = 0
        self.last_move = None
        self.n_moves = 0
        
    def reset_score(self):
        self.score = 0

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        return self.name == other.name

class Board(object):
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.board = np.zeros((2, height, width))
        self.n_marks = 0
    
    def get_state(self):
        return dcopy(self.board)
    
    def in_bounds(self, x, y):
        return x >= 0 and x < self.width and y >= 0 and y < self.height
    
    def is_empty(self, x, y):
        return self.board[0][x][y] == 0 and self.board[1][x][y] == 0
    
    def set(self, id, x, y, value = 1):
        self.board[id][x][y] = value
        if value != 0:
            self.n_marks += 1
            
    def is_fully(self):
        return self.n_marks == self.height * self.width
    
    def to_opp(self):
        self.board = np.flip(self.board, 0)
        return self
    
    def reset(self):
        self.board = np.zeros((2, self.height, self.width))
                              
    def string_representation(self):
        return hash(str(self.board))
    
    def flatten(self):
        self.board.reshape(-1, )
        
    def copy(self):
        return dcopy(self)
    
    def root90(self, k = 0):
        state = dcopy(self)
        state.board[0] = np.rot90(state.board[0], k)
        state.board[1] = np.rot90(self.board[1], k)
        return state
    
    def fliplr(self):
        state = dcopy(self)
        state.board[0] = state.board[0][:, ::-1]
        state.board[1] = self.board[1][:, ::-1]
        return state
    
    def log(self, flip = False, icon = ('X', 'O')):
        if flip:
            icon = (icon[1], icon[0])
        for i in range(self.height):
            row = ['-'] * self.width
            for j in range(self.width):
                if self.board[0][i][j] == 1:
                    row[j] = icon[0]
                elif self.board[1][i][j] == 1:
                    row[j] = icon[1]
            print(row)
        print('-----------')
        
class Environment(object):

    def __init__(self, args):
        self.args = args
        self.show_screen = args.show_screen
        self.n_inputs = 2
        self.max_n_agents = 1
        self.height = args.height
        self.width = args.width
        self.max_n_turns = self.height * self.width
        self.n_actions = self.max_n_turns
        self.num_players = 2
        self.players = [Player(i) for i in range(self.num_players)]
        if args.show_screen:
            self.screen = Screen(self)
            self.screen.init()
    
    """ run in a screen """
    def play(self):
        if not self.args.show_screen:
            raise ValueError('Screen mode unable!')
        self.screen.start()
    
    def get_new_board(self):
        return Board(self.height, self.width)
    
    def reset(self):
        self.players[0].reset_score()
        self.players[1].reset_score()
        self.restart()
            
    def restart(self):
        if self.show_screen:
            self.screen.reset()
        
    """ return a reward after implement action """
    def get_game_ended(self, board, action):
        action = self.convert_action_i2xy(action) 
        if self.check_game_ended(board, 1, action):
            return (True, -1)
        
        if board.is_fully():
            return (True, 0)
        
        return (False, 0)
    
    def render(self):
        """
        display game screen
        """
        if self.args.show_screen:
            self.screen.render()
        
    def get_ub_board_size(self):
        """
        Returns upperbound size of board 
        """
        return [self.height, self.width]
    
    def get_symmetric(self, board, pi):
        sym_boards = []
        sym_pis = []
        pi = np.array(pi).reshape(self.height, self.width)
        for k in [1, 2, 3, 4]:
            b = board.root90(k)
            p = np.rot90(pi, k)
            sym_boards.append(b)
            sym_pis.append(flatten(p))
            sym_boards.append(b.copy().fliplr())
            sym_pis.append(flatten(np.fliplr(p)))
        return sym_boards, sym_pis
    
    def get_states_for_step(self, states):
        states = np.array(states, dtype = np.float32)\
            .reshape(-1, self.n_inputs, self.height, self.width)
        return states
    
    def is_valid_move(self, board, x, y):
        # check if action is valid
        if board.in_bounds(x, y) and board.is_empty(x, y):
            return True
    
    def get_valid_moves(self, board):
        # return a fixed size binary vector
        valids = [0] * self.n_actions
        board = board.get_state()
        for h in range(self.height):
            for w in range(self.width):
                if board[0][h][w] + board[1][h][w] == 0:
                    valids[h * self.width + w] = 1
        return np.array(valids)
    
    def in_board(self, x1 = 0, y1 = 0, x2 = 0, y2 = 0):
        # check if the point is in board
        if x1 < 0 or x1 >= self.height or y1 < 0 or y1 >= self.width:
            return False
        if x2 < 0 or x2 >= self.height or y2 < 0 or y2 >= self.width:
            return False
        return True
    
    def check_game_ended(self, board, playerID, action):
        # check if the game is ended
        x, y = action
        dx = [1, 1, 1, 0, -1, -1, -1, 0]
        dy = [1, 0, -1, -1, -1, 0, 1, 1]
        
        for i in range(4):
            x1, x2, y1, y2 = x, x, y, y
            n_in_rows = -1
            
            while self.in_board(x1, y1) and board.get_state()[playerID][x1][y1] == 1:
                x1 += dx[i]
                y1 += dy[i]
                n_in_rows += 1
                
            while self.in_board(x2, y2) and board.get_state()[playerID][x2][y2] == 1:
                x2 += dx[i + 4]
                y2 += dy[i + 4]
                n_in_rows += 1
            
            if n_in_rows > self.args.n_in_rows:
                return True
            
            if n_in_rows == self.args.n_in_rows:
                if not self.in_board(x1, y1, x2, y2):
                    return True
                if board.get_state()[1 - playerID][x1][y1] == 0 or board.get_state()[1 - playerID][x2][y2] == 0:
                    return True
        return False
    
    # convert action vector to coordinate
    def convert_action_v2c(self, action):
        if not isinstance(action, list):
            raise ValueError('Action must be list!')
        for i in range(len(action)):
            if action[i] == 1:
                return [int(i / self.width), i % self.width]
            
    # convert action vector to int value
    def convert_action_v2i(self, action):
        return np.argmax(action)
            
    # convert action from int to coordinate
    def convert_action_i2xy(self, action):
        if isinstance(action, np.int64) or isinstance(action, int) or isinstance(action, np.int32):
            return (int(action / self.width), action % self.width)
        else:
            raise ValueError('Action must be numeric!')
    
    # convert action from coordinate to int
    def convert_action_c2i(self, action):
        # action need to be a tuple or two element list
        if isinstance(action, list):
            action = tuple(action)
        if isinstance(action, tuple):
            return action[0] * self.width + action[1]
        else:
            raise ValueError('Action must be a tuple or two element list')
    
    # convert action from int to vector
    def convert_action_i2v(self, action):
        # check action is numeric or not
        if not isinstance(action, int):
            raise ValueError('Action must be numeric!')
        _action =  [0] * self.n_actions
        _action[action] = 1
        return _action
    
    # get next state after implement action
    def get_next_state(self, board, action, playerID=None, render=False):
        # convert action from int to coordinate
        if isinstance(action, np.int64) or isinstance(action, int) or isinstance(action, np.int32):
            action = self.convert_action_i2xy(action)
            
        board = board.copy()
        x, y = action
        board.set(0, x, y)
        
        if render: # display on screen
            assert(playerID != None)
            self.screen.draw(x, y, playerID)
            self.screen.render()
            
        return board.to_opp()
            