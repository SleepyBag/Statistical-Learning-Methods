import numpy as np
import sys
import os
from pathlib import Path
from itertools import chain
sys.path.append(str(Path(os.path.abspath(__file__)).parent.parent))
from utils import *

def lda(texts, word_prior_cnt=None, topic_prior_cnt=None, k=5, max_iteration=1000, epsilon=1e-8):
    """
    given a list of token lists, tokens are integers from [0, n_word].
    return the topic distribution of each document,
    and the word distribution of each topic.
    """
    n_word = max(chain(*texts)) + 1
    n_text = len(texts)

    n_text_topic = np.zeros([n_text, k]) + epsilon
    n_topic_word = np.zeros([k, n_word]) + epsilon
    if topic_prior_cnt is not None:
        n_text_topic += topic_prior_cnt[None, :]
    if word_prior_cnt is not None:
        n_topic_word += word_prior_cnt[None, :]

    topic = [[np.random.choice(k) for word in text] for text in texts]
    for i, (text, text_topic) in enumerate(zip(texts, topic)):
        for word, word_topic in zip(text, text_topic):
            n_text_topic[i, word_topic] += 1
            n_topic_word[word_topic, word] += 1

    for step in range(max_iteration):
        for i, (text, text_topic) in enumerate(zip(texts, topic)):
            for j, (word, word_topic) in enumerate(zip(text, text_topic)):
                # reduce the current value from the count
                n_text_topic[i, word_topic] -= 1
                n_topic_word[word_topic, word] -= 1
                # infer the current value from count of others
                likelihood_word_topic = n_topic_word[:, word] / n_topic_word.sum(axis=-1)
                likelihood_topic = n_text_topic[i, :] / n_text_topic[i, :].sum(axis=-1)
                likelihood_topic *= likelihood_word_topic
                p_topic = likelihood_topic / likelihood_topic.sum()
                # update count
                topic[i][j] = np.random.choice(k, p=p_topic)
                n_text_topic[i, topic[i][j]] += 1
                n_topic_word[topic[i][j], word] += 1

    p_topic_when_text = n_text_topic / n_text_topic.sum(axis=-1, keepdims=True)
    p_word_when_topic = n_topic_word / n_topic_word.sum(axis=-1, keepdims=True)
    return p_topic_when_text, p_word_when_topic

if __name__ == '__main__':
    def demonstrate(X, k, desc, **args):
        print(desc)
        p_topic_when_text, p_word_when_topic = lda(X, k=k, **args)
        print("The probabilities of each topic for each text are")
        print(np.round(p_topic_when_text, 2))
        print("The probabilities of each word for each topic are")
        print(np.round(p_word_when_topic, 2))
        print("The recovered text-wordcnt matrix is")
        print(np.round((p_topic_when_text @ p_word_when_topic), 2))
        print()

    n_vocab = 9
    X = [
        [2, 3],
        [5, 8],
        [1, 7],
        [6, 8],
        [0, 5],
        [0, 1, 2, 3, 4, 5, 6, 7, 8],
        [0, 2],
        [6, 8],
        [5, 5, 8],
        [0, 2, 7],
        [3, 4]
    ]
    demonstrate(X, 3, 'Example 1')
    demonstrate(X, 8, 'Example 2: You can recogonize the original matrix from the recovered one if k is large enough')

    k = 8
    word_prior_cnt = np.ones(n_vocab) * 2
    topic_prior_cnt = np.ones(k) * 2
    demonstrate(X, k, 'Example 3: The influence of prior', word_prior_cnt=word_prior_cnt, topic_prior_cnt=topic_prior_cnt)

    k = 8
    word_prior_cnt = np.ones(n_vocab) * 2
    topic_prior_cnt = np.zeros(k)
    topic_prior_cnt[3] = 5
    demonstrate(X, k, 'Example 4: The influence of prior', word_prior_cnt=word_prior_cnt, topic_prior_cnt=topic_prior_cnt)
