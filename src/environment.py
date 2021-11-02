from copy import deepcopy as dcopy
from numpy import array
from GameBoard.game_board import Screen
from src.utils import flatten
import numpy as np

class Player(object):
    
    def __init__(self, ID):
        self.ID = ID
        self.n_wins = 0
        
    def reset(self):
        self.n_wins = 0
        
class Board(object):
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.board = np.zeros((2, height, width))
    
    def get_state(self):
        return self.board
    
    def set(self, id, x, y, value):
        self.board[id][x][y] = value
    
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
        self.board[0] = np.rot90(self.board[0], k)
        self.board[1] = np.rot90(self.board[1], k)
    
    def fliplr(self):
        self.board[0] = self.board[0][:, ::-1]
        self.board[1] = self.board[1][:, ::-1]
    
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
        self.board = Board(args.height, args.width)
        self.max_n_turns = self.height * self.width
        self.n_actions = self.max_n_turns
        self.num_players = 2
        self.players = [Player(i) for i in range(self.num_players)]
        self.screen = Screen(self)
        self.reset()
    
    """ run in a screen """
    def play(self):
        if not self.args.show_screen:
            raise ValueError('Screen mode unable!')
        self.screen.start()
    
    def reset(self):
        self.board.reset()
        
        if self.show_screen:
            self.screen.setup(self)
        
    """ return a reward after implement action """
    def get_game_ended(self, board, action):
        action = self.convert_action_i2c(action) 
        if self.check_game_ended(board, 1, action):
            return (True, 1)
        
        if np.sum(board.get_state()) == self.n_actions:
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
        sym_board = board.copy()
        sym_pi = np.array(pi).reshape(self.height, self.width)
        if np.random.choice([True, False]):
            k =  np.random.randint(4)
            sym_board.root90(k)
            sym_pi = np.rot90(sym_pi, k)
            
        if np.random.choice([True, False]):
            sym_board.fliplr()
            sym_pi = np.fliplr(sym_pi)
        sym_pi = flatten(sym_pi)
        return sym_board, sym_pi
    
    def get_states_for_step(self, states):
        states = np.array(states, dtype = np.float32)\
            .reshape(-1, self.n_inputs, self.height, self.width)
        return states
    
    def get_valid_moves(self, board):
        # return a fixed size binary vector
        valids = [0] * self.n_actions
        _board = board.get_state()
        for h in range(self.height):
            for w in range(self.width):
                if _board[0][h][w] + _board[1][h][w] == 0:
                    valids[h * self.width + w] = 1
        return np.array(valids)
    
    def is_outside(self, x, y):
        return x < 0 or x >= self.width or y < 0 or y >= self.height
    
    def is_inside(self, x, y):
        return x >= 0 and x < self.width and y >= 0 and y < self.height
    
    def check_game_ended(self, board, playerID, action):
        n_rows = self.args.n_rows
        x, y = action
        
        dx = [0, 1, 0, -1]
        dy = [1, 0, -1, 0]
        
        for i in range(4):
            x1, x2, y1, y2 = x, x, y, y
            while self.is_inside(x1, y) and board.get_state()[playerID][x1][y] == 1:
                x1 -= 1
                horizontal += 1
                
            while self.is_inside(x2, y) and board.get_state()[playerID][x2][y] == 1:
                x2 += 1
                horizontal += 1
                
            
            
        
        horizontal = -1
        
        """ HORIZONTAL """
        if horizontal > n_rows:
            return True
        
        if x1 < 0 or x2 >= self.width:
            if horizontal == n_rows:
                return True
        else:
            if horizontal == n_rows and (board.get_state()[1 - playerID][x1][y] == 0 or board.get_state()[1 - playerID][x2][y] == 0):
                return True
            
        """ VERTICAL """
        x1, x2, y1, y2 = x, x, y, y
        vertical = -1
        
        while self.is_inside(x, y1) and board.get_state()[playerID][x][y1] == 1:
            y1 -= 1
            vertical += 1
            
        while self.is_inside(x, y2) and board.get_state()[playerID][x][y2] == 1:
            y2 += 1
            vertical += 1
            
        if vertical > n_rows:
            return True
        
        if y1 < 0 or y2 >= self.height:
            if vertical == n_rows:
                return True
        else:
            if vertical == n_rows and (board.get_state()[1 - playerID][x][y1] == 0 or board.get_state()[1 - playerID][x][y2] == 0):
                return True
        
        """ DIAGONAL \ """
        x1, x2, y1, y2 = x, x, y, y
        diagonal = -1
        
        while self.is_inside(x1, y1) and board.get_state()[playerID][x1][y1] == 1:
            x1 -= 1
            y1 -= 1
            diagonal += 1
            
        while self.is_inside(x2, y2) and board.get_state()[playerID][x2][y2] == 1:
            x2 += 1
            y2 += 1
            diagonal += 1
            
        if diagonal > n_rows:
            return True
        
        if diagonal == n_rows:
            if (self.is_outside(x1, y1)) or (self.is_outside(x2, y2)):
                return True
            if (board.get_state()[1 - playerID][x1][y1] == 0 or board.get_state()[1 - playerID][x2][y2] == 0):
                return True            
            
        """ DIAGONAL / """
        x1, x2, y1, y2 = x, x, y, y
        diagonal = -1
        
        while self.is_inside(x1, y1) and board.get_state()[playerID][x1][y1] == 1:
            x1 -= 1
            y1 += 1
            diagonal += 1
            
        while self.is_inside(x2, y2) and board.get_state()[playerID][x2][y2] == 1:
            x2 += 1
            y2 -= 1
            diagonal += 1
            
        if diagonal > n_rows:
            return True
        
        if diagonal == n_rows:
            if (self.is_outside(x1, y1)) or (self.is_outside(x2, y2)):
                return True
            if (board.get_state()[1 - playerID][x1][y1] == 0 or board.get_state()[1 - playerID][x2][y2] == 0):
                return True
        
        return False
            
    def convert_action_v2c(self, action):
        for i in range(action):
            if action[i] == 1:
                return [int(i / self.width), i % self.width]
            
    # convert action from int to coordinate
    def convert_action_i2c(self, action):
        return [int(action / self.width), action % self.width]
    
    # convert action from coordinate to int
    def convert_action_c2i(self, action):
        return action[0] * self.width + action[1]
    
    # convert action from int to vector
    def convert_action_i2v(self, action):
        _action =  [0] * self.n_actions
        _action[action] = 1
        return _action
    
    # get next state after implement action
    def get_next_state(self, board, action, playerID = None, render = False):
        action = self.convert_action_i2c(action)        
        x, y = action
        board = board.copy()
        board.set(0, x, y, 1)
        
        if render: # display on screen
            assert(playerID != None)
            self.screen.draw(x, y, playerID)
            self.screen.render()
            
        return board.to_opp()
            