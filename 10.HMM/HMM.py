import os
import numpy as np
from functools import partial
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
sys.path.append(str(Path(os.path.abspath(__file__)).parent.parent))
from utils import *
sys.path.append(str(Path(os.path.abspath(__file__)).parent.parent / '10.HMM'))
from BaumWelch import baum_welch
from Viterbi import viterbi

class HMM:
    def __init__(self, state_size, observation_size, max_iteration=2000, verbose=False, epsilon=1e-8):
        self.max_iteration = max_iteration
        self.verbose = verbose
        self.state_size = state_size
        self.observation_size = observation_size
        self.epsilon = epsilon

    def fit(self, X):
        """
        When there is no label in the training data,
        HMM uses baum-welch for training.
        Otherwise just counting the probability will be fine (not implemented here)
        """
        self.state2state, self.state2observation, self.initial_state = \
            baum_welch(X, self.state_size, self.observation_size, self.epsilon, self.max_iteration)

    def predict(self, X):
        """HMM uses viterbi for predicting"""
        Y = np.zeros_like(X)
        Y = np.apply_along_axis(
            partial(viterbi, self.state2state, self.state2observation, self.initial_state), -1, X)
        return Y


if __name__ == '__main__':
    def demonstrate(X, Y, desc):
        console = Console(markup=False)

        vocab = set(X.flatten())
        vocab_size = len(vocab)
        word2num = {word: num for num, word in enumerate(vocab)}

        f_word2num = np.vectorize(lambda word: word2num[word])

        numX, numY = map(f_word2num, (X, Y))

        hmm = HMM(4, vocab_size)
        hmm.fit(numX)
        pred = hmm.predict(numY)

        # show in table
        print(desc)
        table = Table()
        for y, p in zip(Y, pred):
            table.add_row(*map(str, y))
            table.add_row(*map(str, p))
        console.print(table)


    # ---------------------- Example 1 --------------------------------------------
    X = np.array([s.split() for s in
                  ['i am good .',
                   'i am bad .',
                   'you are good .',
                   'you are bad .',
                   'it is good .',
                   'it is bad .',
                   ]
                  ])
    Y = X
    demonstrate(X, Y, "Example 1")

    # ---------------------- Example 2 --------------------------------------------
    Y = np.array([s.split() for s in
                  ['you is good .',
                   'i are bad .',
                   'it are good .']
                  ])
    Y = np.concatenate([X, Y])
    demonstrate(X, Y, "Example 2")
