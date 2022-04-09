from importlib.metadata import entry_points
from setuptools import setup
from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

setup (
    name = 'MonteCarloTreeSearch-Gomoku',
    version = '0.1',
    description = 'Monte Carlo Tree Search for Gomoku',
    long_description = long_description,
    long_description_content_type="text/markdown", 
    author = 'Vu Quoc Hien',
    author_email = 'hienvq.2000@gmail.com',
    url = 'https://bitbucket.org/Hienvq2304/deep_hla',
    packages=["src", "models", "GameBoard"],
    keywords='gomoku, deep reinforcement learning, monte carlo tree search',
    install_requires=[
        'torch==1.9.0',
        'scikit-learn==0.24.2',
        'sklearn==0.0',
        'tqdm==4.62.2',
        'matplotlib==3.4.3',
        'pygame==2.0.1'
    ],
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'gomoku=mcts_training:main'
        ]
    },
    
)