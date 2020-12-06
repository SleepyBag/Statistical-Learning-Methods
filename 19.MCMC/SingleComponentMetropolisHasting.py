import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde


def gaussian_kernel(x, j, xj_new):
    return np.exp(-(x[j] - xj_new) ** 2)

def gaussian_sampler(x, j):
    return np.random.normal(x[j])

def single_component_metropolis_hasting(dim, p, q=gaussian_kernel, q_sampler=gaussian_sampler, x0=None, burning_steps=1000, max_steps=10000, epsilon=1e-8, verbose=False):
    """
    Given a distribution function p (it doesn't need to be a probability, a likelihood function is enough),
    and the recommended distribution q,
    return a list of samples x ~ p,
    where the number of samples is max_steps - burning_steps.
    q_sampler is a function taking an (x, j) as input and return a sample of q(xj_new | xj_old, old_x_without_xj)
    q is a distribution function representing q(xj_new, xj_old | old_x_without_xj).
    q takes (x, j, xj_new) as parameters,
    where x is the variable last step,
    j is index of the the parameter chosen to be updated,
    xj_new is the new value of x_j.
    x0 is the initial value of x. If not specified, it's set as zero vector.
    """
    x = np.zeros(dim)
    samples = np.zeros([max_steps - burning_steps, dim])
    # Burning
    for i in range(max_steps):
        for j in range(dim):
            xj_new = q_sampler(x, j)
            x_new = x.copy()
            x_new[j] = xj_new
            accept_prob = (p(x_new) + epsilon) / (p(x) + epsilon) * q(x, j, xj_new) / q(x_new, j, x[j])
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
        samples = single_component_metropolis_hasting(dim, p, **args)
        z = gaussian_kde(samples.T)(samples.T)
        plt.scatter(samples[:, 0], samples[:, 1], c=z, marker='.')
        plt.plot(samples[: 100, 0], samples[: 100, 1], 'r-')
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
    demonstrate(2, blocks, "Gaussian distribution with mean of 2 and 3")
