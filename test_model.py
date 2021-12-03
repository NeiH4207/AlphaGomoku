from numpy.__config__ import show
from src.environment import Environment

from src.utils import dotdict
from src.model import Policy
from src.SimpleNet2 import GomokuNet
from src.evaluate import Evaluation
from random import seed

def main():
    
    args = dotdict({
        'height': 3,
        'width': 3,
        "n_in_rows": 3,
        'depth_minimax': 3,
        'show_screen': True,
        'speed': 0.5,
        'num_iters': 1000,
        'num_epochs': 50,
        'nCompare': 100,
        'mem_size': 10000,
        'mode': 'test-machine',
        'saved_model': False ,
        'load_folder_file_1': ('Models','nnet.pt'),
        'load_folder_file_2': ('Models','rejected_nnet.pt')
    })

    env = Environment(args)
    nnet = GomokuNet(env)
    nnet.load_checkpoint(args.load_folder_file_1[0], args.load_folder_file_1[1])
    pnet = GomokuNet(env)
    # pnet.load_checkpoint(args.load_folder_file_2[0], args.load_folder_file_2[1])

    print('OLD ELO: {} / {}'.format(nnet.elo, pnet.elo))
    eval = Evaluation(env, nnet, pnet)
    eval.run()
    
    nwins, pwins, draws = eval.get_info()

    print('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
    print('NEW ELO: {} / {}'.format(nnet.elo, pnet.elo))
    
if __name__ == "__main__":
    main()