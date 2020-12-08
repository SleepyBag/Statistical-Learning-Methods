import numpy as np
import sys
import os
from pathlib import Path
sys.path.append(str(Path(os.path.abspath(__file__)).parent.parent))
from utils import *

def plsa(word_text, k=5, max_iteration=1000, epsilon=1e-8):
    """
    given a word-text matrix
    the dimension of the principle component, k
    optimize using EM algorithm
    return the word-topic matrix and text-topic matrix
    """
    n_word, n_text = word_text.shape
    p_topic_when_text = np.random.rand(n_text, k)
    p_word_when_topic = np.random.rand(k, n_word)

    text_word = word_text.T
    text_word_cnt = text_word.sum(axis=-1, keepdims=True)
    for i in range(max_iteration):
        # E step: calculate the expectation of each topic for each word-text pair
        p_topic_when_text_word = p_topic_when_text[:, :, None] * p_word_when_topic[None, :, :]
        p_topic_when_text_word /= p_topic_when_text_word.sum(axis=1, keepdims=True) + epsilon

        # M step, maximazation the likelihood of the observation, i.e., the word-text matrix
        topic_cnt = text_word[:, None, :] * p_topic_when_text_word
        p_word_when_topic = (topic_cnt).sum(axis=0) / \
            (topic_cnt).sum(axis=0).sum(axis=-1, keepdims=True)
        p_topic_when_text = (text_word[:, None, :] * p_topic_when_text_word).sum(axis=-1) / text_word_cnt
    return p_topic_when_text, p_word_when_topic

if __name__ == '__main__':
    def demonstrate(X, k, desc):
        print(desc)
        p_topic_when_text, p_word_when_topic = plsa(X, k=k)
        print("The probabilities of each topic for each text are")
        print(np.round(p_topic_when_text, 2))
        print("The probabilities of each word for each topic are")
        print(np.round(p_word_when_topic, 2))
        print("The recovered text-wordcnt matrix is")
        print(np.round((p_topic_when_text @ p_word_when_topic).T, 2))
        print()

    X = np.array([
        [0, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 1],
        [0, 1, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 2, 0, 0, 1],
        [1, 0, 1, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 1, 0, 0, 0, 0],
    ]).astype(float)
    demonstrate(X, 3, 'Example 1')

    X = np.array([
        [0, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 1],
        [0, 1, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 2, 0, 0, 1],
        [1, 0, 1, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 1, 0, 0, 0, 0],
    ]).astype(float)
    demonstrate(X, max(X.shape), 'Example 2: You can recogonize the original matrix from the recovered one if k is large enough')
