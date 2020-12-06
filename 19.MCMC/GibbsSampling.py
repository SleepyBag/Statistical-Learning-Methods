import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde


def gibbs_sampling(dim, conditional_sampler, x0=None, burning_steps=1000, max_steps=10000, epsilon=1e-8, verbose=False):
    """
    Given a conditionl sampler which samples from p(x_j | x_1, x_2, ... x_n)
    return a list of samples x ~ p, where p is the original distribution of the conditional distribution.
    x0 is the initial value of x. If not specified, it's set as zero vector.
    conditional_sampler takes (x, j) as parameters
    """
    x = np.zeros(dim) if x0 is None else x0
    samples = np.zeros([max_steps - burning_steps, dim])
    for i in range(max_steps):
        for j in range(dim):
            x[j]  = conditional_sampler(x, j)
            if verbose:
                print("New value of x is", x_new)
        if i >= burning_steps:
            samples[i - burning_steps] = x
    return samples


if __name__ == '__main__':
    def demonstrate(dim, p, desc, **args):
        samples = gibbs_sampling(dim, p, **args)
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
    def gaussian_sampler1(x, j):
        return np.random.normal()
    demonstrate(2, gaussian_sampler1, "Gaussian distribution with mean of 0 and 0")

    # example 2:
    mean = np.array([2, 3])
    covariance = np.array([[1, 0],
                           [0, 1]])
    covariance_inv = np.linalg.inv(covariance)
    det_convariance = 1
    def gaussian_sampler2(x, j):
        if j == 0:
            return np.random.normal(2)
        else:
            return np.random.normal(3)
    demonstrate(2, gaussian_sampler2, "Gaussian distribution with mean of 2 and 3")

    # example 3:
    def blocks_sampler(x, j):
        sample = np.random.random()
        if sample > .5:
            sample += 1.
        return sample
    demonstrate(2, blocks_sampler, "Four blocks")

    # example 4:
    def blocks_sampler(x, j):
        sample = np.random.random()
        if sample > .5:
            sample += 100.
        return sample
    demonstrate(2, blocks_sampler, "Four blocks with large gap.")
