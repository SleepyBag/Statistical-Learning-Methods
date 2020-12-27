from math import log
import os
from matplotlib.tri.triinterpolate import LinearTriInterpolator
import numpy as np
from functools import partial
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
sys.path.append(str(Path(os.path.abspath(__file__)).parent.parent))
from utils import *

class LinearChainConditionalRandomField:
    def __init__(self, feature_funcs, trans_feature_funcs, sequence_length, n_x, n_y, max_iteration=100, verbose=False):
        """
        `feature_funcs` are a group of functions s(y_i, X, i) in a list
        `trans_feature_funcs` are a group of functions t(y_{i-1}, y_i, X, i) in a list
        `sequence_length` is the length of each input sequence
        `n_x` is the number of possible values of each item in a sequence x
        `n_y` is the number of possible values of each item in a sequence y
        """
        self.feature_funcs = feature_funcs
        self.trans_feature_funcs = trans_feature_funcs
        self.n_x = n_x
        self.n_y = n_y
        self.sequence_length = sequence_length
        self.max_iteration = max_iteration
        self.verbose = verbose

    def get_trans(self, x):
        """get transition matrix given observed sequence x"""
        trans_feature = np.zeros([self.sequence_length, self.n_y, self.n_y])
        for i in range(self.sequence_length):
            for y_i_1 in range(self.n_y):
                for y_i in range(self.n_y):
                    for j, func in enumerate(self.used_feature_funcs):
                        trans_feature[i, y_i_1, y_i] += self.w_feature_funcs[j] * func(y_i, x, i)
            if i > 0:
                for y_i_1 in range(self.n_y):
                    for y_i in range(self.n_y):
                        for j, func in enumerate(self.used_trans_feature_funcs):
                            trans_feature[i, y_i_1, y_i] += self.w_trans_feature_funcs[j] * func(y_i_1, y_i, x, i)
        return np.exp(trans_feature)

    def fit(self, X, Y):
        """
        X is a two dimensional matrix of observation sequence
        Y is a two dimensional matrix of hidden state sequence
        optimize weights by Improved Iterative Scaling
        """
        E_feature = np.zeros(len(self.feature_funcs))
        E_trans_feature = np.zeros(len(self.trans_feature_funcs))

        # Because each x is a sequence, it's vector space is too large to iterate.
        # We need to store all the possible sequence x during the training time
        # and only iterate over existing x.
        p_x = {tuple(x): 0. for x in X}

        for x, y in zip(X, Y):
            x_key = tuple(x)
            p_x[x_key] += 1 / len(X)
            for i, yi in enumerate(y):
                for j, func in enumerate(self.feature_funcs):
                    E_feature[j] += func(yi, x, i) / len(X)
            for i in range(1, self.sequence_length):
                yi_1, yi = y[i - 1], y[i]
                for j, func in enumerate(self.trans_feature_funcs):
                    E_trans_feature[j] += func(yi_1, yi, x, i) / len(X)

        # features that don't show in training data are useless, filter them
        self.used_feature_funcs = [func for E, func in zip(E_feature, self.feature_funcs) if E != 0]
        self.used_trans_feature_funcs = [func for E, func in zip(E_trans_feature, self.trans_feature_funcs) if E != 0]
        E_feature = E_feature[E_feature.nonzero()]
        E_trans_feature = E_trans_feature[E_trans_feature.nonzero()]
        self.w_feature_funcs = np.zeros(len(self.used_feature_funcs))
        self.w_trans_feature_funcs = np.zeros(len(self.used_trans_feature_funcs))

        # pre-calculate all the possible values of feature functions
        feature = np.zeros([len(self.used_feature_funcs), len(p_x), self.sequence_length, self.n_y])
        trans_feature = np.zeros([len(self.used_trans_feature_funcs), len(p_x), self.sequence_length, self.n_y, self.n_y])
        for x_i, x_key in enumerate(p_x):
            x = np.array(x_key)
            for func_i, func in enumerate(self.used_trans_feature_funcs):
                for i in range(1, self.sequence_length):
                    for y_i_1 in range(self.n_y):
                        for y_i in range(self.n_y):
                            trans_feature[func_i, x_i, i, y_i_1, y_i] = func(y_i_1, y_i, x, i)
            for func_i, func in enumerate(self.used_feature_funcs):
                for i in range(self.sequence_length):
                    for y_i in range(self.n_y):
                        feature[func_i, x_i, i, y_i] = func(y_i, x, i)

        # pre-calculate the max number of features, given x
        max_feature = np.zeros(len(p_x), dtype=int)
        sum_trans_feature = trans_feature.sum(axis=0)
        sum_feature = feature.sum(axis=0)
        for x_i, x_key in enumerate(p_x):
            cur_max_feature = np.zeros(self.n_y)
            for i in range(self.sequence_length):
                cur_max_feature = (cur_max_feature[:, None] + sum_trans_feature[x_i, i]).max(axis=0) + sum_feature[x_i, i]
            max_feature[x_i] = cur_max_feature.max()
        n_coef = max(max_feature) + 1

        # train
        for iteration in range(self.max_iteration):
            if self.verbose:
                print(f'Iteration {iteration} starts...')
            loss = 0.
            for funcs, w, E_experience in zip(
                    [self.used_feature_funcs, self.used_trans_feature_funcs],
                    [self.w_feature_funcs, self.w_trans_feature_funcs],
                    [E_feature, E_trans_feature]):
                for func_i in range(len(funcs)):
                    # if funcs is self.used_trans_feature_funcs:
                    coef = np.zeros(n_coef)
                    # only iterater over possible x
                    for x_i, x_key in enumerate(p_x):
                        cur_p_x = p_x[x_key]
                        x = np.array(x_key)

                        trans = self.get_trans(x)
                        # forward algorithm
                        cur_prob = np.ones(self.n_y)
                        forward_prob = np.zeros([self.sequence_length + 1, self.n_y])
                        forward_prob[0] = cur_prob
                        for i in range(self.sequence_length):
                            cur_prob = cur_prob @ trans[i]
                            forward_prob[i + 1] = cur_prob
                        # backward algorithm
                        cur_prob = np.ones(self.n_y)
                        backward_prob = np.zeros([self.sequence_length + 1, self.n_y])
                        backward_prob[-1] = cur_prob
                        for i in range(self.sequence_length - 1, -1, -1):
                            cur_prob = trans[i] @ cur_prob
                            backward_prob[i] = cur_prob

                        if iteration < 10:
                            np.testing.assert_almost_equal(
                                forward_prob[-1].sum(),
                                backward_prob[0].sum()
                            )
                            for i in range(1, self.sequence_length + 1):
                                np.testing.assert_almost_equal(
                                    forward_prob[i] @ backward_prob[i],
                                    forward_prob[-1].sum()
                                )
                            for i in range(0, self.sequence_length):
                                np.testing.assert_almost_equal(
                                    (np.outer(forward_prob[i], backward_prob[i + 1]) * trans[i]).sum(),
                                    forward_prob[-1].sum()
                                )

                        # calculate expectation of each feature_function given x
                        cur_E_feature = 0.
                        if funcs is self.used_feature_funcs:
                            for i in range(1, self.sequence_length + 1):
                                cur_E_feature += (
                                    forward_prob[i] * backward_prob[i] * feature[func_i, x_i, i - 1]
                                ).sum()
                        elif funcs is self.used_trans_feature_funcs:
                            for i in range(0, self.sequence_length):
                                cur_E_feature += (
                                    np.outer(forward_prob[i], backward_prob[i + 1]) * trans[i] * trans_feature[func_i, x_i, i]
                                ).sum()
                        else:
                            raise Exception("Unknown function set!")
                        cur_E_feature /= forward_prob[-1].sum()

                        coef[max_feature[x_i]] += cur_p_x * cur_E_feature

                    # update w
                    dw_i = log(newton(
                        lambda x: sum(c * x ** i for i, c in enumerate(coef)) - E_experience[func_i],
                        lambda x: sum(i * c * x ** (i  - 1) for i, c in enumerate(coef) if i > 0),
                        1
                    ))
                    w[func_i] += dw_i
                    loss += abs(E_experience[func_i] - coef.sum())
            loss /= len(self.feature_funcs) + len(self.trans_feature_funcs)
            if self.verbose:
                print(f'Iteration {iteration} ends, Loss: {loss}')

    def predict(self, X):
        """
        predict state sequence y using viterbi algorithm
        X is a group of sequence x in a two-dimensional array
        """

        ans = np.zeros([len(X), self.sequence_length])
        for x_i, x in enumerate(X):
            # pre-calculate all the possible values of feature functions
            feature = np.zeros([len(self.used_feature_funcs), self.sequence_length, self.n_y])
            trans_feature = np.zeros([len(self.used_trans_feature_funcs), self.sequence_length, self.n_y, self.n_y])
            for func_i, func in enumerate(self.used_trans_feature_funcs):
                for i in range(1, self.sequence_length):
                    for y_i_1 in range(self.n_y):
                        for y_i in range(self.n_y):
                            trans_feature[func_i, i, y_i_1, y_i] = func(y_i_1, y_i, x, i)
            for func_i, func in enumerate(self.used_feature_funcs):
                for i in range(self.sequence_length):
                    for y_i in range(self.n_y):
                        feature[func_i, i, y_i] = func(y_i, x, i)
            feature = (self.w_feature_funcs[:, None, None] * feature).sum(axis=0)
            trans_feature = (self.w_trans_feature_funcs[:, None, None, None] * trans_feature).sum(axis=0)

            # viterbi
            pre_state = np.zeros([self.sequence_length, self.n_y], dtype=int) - 1
            prob = np.zeros([self.sequence_length, self.n_y])
            cur_prob = np.ones(self.n_y)
            for i in range(self.sequence_length):
                trans_prob = cur_prob[:, None] + trans_feature[i]
                pre_state[i] = trans_prob.argmax(axis=0)
                cur_prob = trans_prob.max(axis=0) + feature[i]
                prob[i] = cur_prob

            # back track the trace
            cur_state = prob[-1].argmax()
            for i in range(self.sequence_length - 1, -1, -1):
                ans[x_i, i] = cur_state
                cur_state = pre_state[i, cur_state]
        return ans


if __name__ == '__main__':
    def demonstrate(X, Y, testX, n_y, desc):
        console = Console(markup=False)

        vocab = set(X.flatten())
        vocab_size = len(vocab)
        word2num = {word: num for num, word in enumerate(vocab)}

        f_word2num = np.vectorize(lambda word: word2num[word])

        numX, num_testX = map(f_word2num, (X, testX))

        sequence_length = numX.shape[-1]

        class FeatureFunc:
            def __init__(self, x_i, y_i):
                self.x_i = x_i
                self.y_i = y_i

            def __call__(self, y_i, x, i):
                return int(y_i == self.y_i and x[i] == self.x_i)

        class TransFeatureFunc:
            def __init__(self, y_i_1, y_i):
                self.y_i = y_i
                self.y_i_1 = y_i_1

            def __call__(self, y_i_1, y_i, x, i):
                return int(y_i_1 == self.y_i_1 and y_i == self.y_i)

        feature_funcs = [FeatureFunc(x_i, y_i)
                         for x_i in range(vocab_size)
                         for y_i in range(n_y)]
        trans_feature_funcs = [TransFeatureFunc(y_i_1, y_i)
                               for y_i_1 in range(n_y)
                               for y_i in range(n_y)]

        linear_chain_conditional_random_field = LinearChainConditionalRandomField(
            feature_funcs,
            trans_feature_funcs,
            sequence_length,
            vocab_size,
            n_y,
            verbose=True
        )
        linear_chain_conditional_random_field.fit(numX, Y)
        pred = linear_chain_conditional_random_field.predict(num_testX)

        # show in table
        print(desc)
        table = Table()
        for x, p in zip(testX, pred):
            table.add_row(*map(str, x))
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
    Y = np.array([
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        [0, 1, 2, 3],
    ])
    testX = np.array([s.split() for s in
                  ['you is good .',
                   'i are bad .',
                   'it are good .']
                  ])
    testX = np.concatenate([X, testX])
    demonstrate(X, Y, testX, 4, "Example 1")

    # ---------------------- Example 1 --------------------------------------------
    X = np.array([s.split() for s in
                  ['i be good .',
                   'you be good .',
                   'be good . .',
                   'i love you .',
                   'he be . .',
                   ]
                  ])
    # pronoun: 0, verb: 1, adjective: 2, ".": 3
    Y = np.array([
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        [1, 2, 3, 3],
        [0, 1, 0, 3],
        [0, 1, 3, 3],
    ])
    testX = np.array([s.split() for s in
                  ['you be good .',
                   'he love you .',
                   'i love good .',
                   '. be love .',
                   '. love be .',
                   '. . be good']
                  ])
    testX = np.concatenate([X, testX])
    demonstrate(X, Y, testX, 4, "Example 2")
