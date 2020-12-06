import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde


def gaussian_kernel(x1, x2):
    return np.exp(-((x1 - x2) ** 2).sum())

def gaussian_sampler(x):
    return np.random.normal(x)

def metropolis_hasting(dim, p, q=gaussian_kernel, q_sampler=gaussian_sampler, x0=None, burning_steps=1000, max_steps=10000, epsilon=1e-8, verbose=False):
    """
    Given a distribution function p (it doesn't need to be a probability, a likelihood function is enough),
    and the recommended distribution q,
    return a list of samples x ~ p,
    where the number of samples is max_steps - burning_steps.
    q_sampler is a function taking an x as input and return a sample of q(x_new | x_old).
    q is a distribution function representing q(x_new | x_old).
    q takes (x_old, x_new) as parameters.
    """
    x = np.zeros(dim) if x0 is None else x0
    samples = np.zeros([max_steps - burning_steps, dim])
    for i in range(max_steps):
        x_new = q_sampler(x)
        accept_prob = (p(x_new) + epsilon) / (p(x) + epsilon) * q(x, x_new) / q(x_new, x)
        if verbose:
            print("New value of x is", x_new)
        if np.random.random() < accept_prob:
            x = x_new
        elif verbose:
            print("New value is dropped")
        if i >= burning_steps:
            samples[i - burning_steps] = x
    return samples


if __name__ == '__main__':
    def demonstrate(dim, p, desc, **args):
        samples = metropolis_hasting(dim, p, **args)
        z = gaussian_kde(samples.T)(samples.T)
        plt.scatter(samples[:, 0], samples[:, 1], c=z, marker='.')
        plt.plot(samples[: 100, 0], samples[: 100, 1], 'r-')
        plt.title(desc)
        plt.show()

    # example 1:
    mean = np.array([2, 3])
    covariance = np.array([[1, 0],
                           [0, 1]])
    covariance_inv = np.linalg.inv(covariance)
    det_convariance = 1
    def gaussian1(x):
        return np.exp(-.5 * (x - mean).T @ covariance_inv @ (x - mean))
    demonstrate(2, gaussian1, "Gaussian distribution with mean of 2 and 3")

    # example 2:
    mean = np.array([2, 3])
    covariance = np.array([[1, .5],
                           [.5, 1]])
    covariance_inv = np.linalg.inv(covariance)
    det_convariance = 1
    def gaussian2(x):
        return np.exp(-.5 * (x - mean).T @ covariance_inv @ (x - mean))
    demonstrate(2, gaussian2, "Gaussian distribution with mean of 2 and 3")

    # example 3:
    def blocks(x):
        if (0 < x[0] < 1 or 2 < x[0] < 3) and (0 < x[1] < 1 or 2 < x[1] < 3):
            return 1
        return 0
    demonstrate(2, blocks, "Four blocks")

    # example 4:
    def blocks(x):
        if (0 < x[0] < 1 or 200 < x[0] < 300) and (0 < x[1] < 1 or 200 < x[1] < 300):
            return 1
        return 0
    demonstrate(2, blocks, "Four blocks with large gap. (Monte Carlo doesn't solve everything)")
