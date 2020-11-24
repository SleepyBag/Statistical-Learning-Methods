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
    return the word-latent matrix and text-latent matrix
    """
    n_word, n_text = word_text.shape
    word_latent = np.random.rand(n_word, k)
    latent_text = np.random.rand(k, n_text)
    for i in range(max_iteration):
        word_latent *= (word_text @ latent_text.T) / (word_latent @ latent_text @ latent_text.T)
        latent_text *= (word_latent.T @ word_text) / (word_latent.T @ word_latent @ latent_text)
    return word_latent, latent_text.T

if __name__ == '__main__':
    def demonstrate(X, k, desc):
        print(desc)
        word_latent, text_latent = lsa(X, k=k)
        print("The latent vectors of all the words are")
        print(word_latent)
        print("The latent vectors of all the texts are")
        print(text_latent)
        print("The recovered word-text matrix is")
        print(np.round(word_latent @ text_latent.T))

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
