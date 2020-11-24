import numpy as np
import sys
import os
from pathlib import Path
sys.path.append(str(Path(os.path.abspath(__file__)).parent.parent))
from utils import *

def lsa(word_text, k=5, max_iteration=1000):
    """
    given a word-text matrix
    the dimension of the principle component, k
    optimize using the algorithm proposed by Lee and Seung
    return the word-topic matrix and text-topic matrix
    """
    n_word, n_text = word_text.shape
    word_topic = np.random.rand(n_word, k)
    topic_text = np.random.rand(k, n_text)
    for i in range(max_iteration):
        word_topic *= (word_text @ topic_text.T) / (word_topic @ topic_text @ topic_text.T)
        topic_text *= (word_topic.T @ word_text) / (word_topic.T @ word_topic @ topic_text)
    return word_topic, topic_text.T

if __name__ == '__main__':
    def demonstrate(X, k, desc):
        print(desc)
        word_topic, text_topic = lsa(X, k=k)
        print("The topic vectors of all the words are")
        print(word_topic)
        print("The topic vectors of all the texts are")
        print(text_topic)
        print("The recovered word-text matrix is")
        print(np.round(word_topic @ text_topic.T))

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
