from matplotlib import pyplot as plt
import numpy as np

def kbline(k, b, **args):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = b + k * x_vals
    plt.plot(x_vals, y_vals, **args)

def wbline(w, b, **args):
    k = -w[0] / w[1]
    b /= -w[1]
    if np.isinf(k):
        plt.vlines(b / w1, plt.gca().get_ylim(), **args)
    else:
        kbline(k, b, **args)
