import numpy as np
from matplotlib import pyplot as plt

class GMM:
    def __init__(self, k, max_step=200):
        self.k = k
        self.max_step = max_step
        self.epsilon = 1e-8

    def fit(self, X):
        """
        X: training data of shape [n, feature_size]
        """
        n, feature_size = X.shape
        # the parameter of each gaussian distribution
        self.prior = np.ones(self.k) / self.k
        self.prior /= self.prior.sum()
        self.mean = np.random.rand(self.k, feature_size)
        self.std = np.std(X, axis=0, keepdims=True)

        for step in range(self.max_step):
            # Expectation step
            posterior = self.predict(X)

            # Maximum step
            self.mean = (posterior[:, :, None] * X[None, :, :]).sum(axis=1) / \
                (posterior.sum(axis=1)[:, None] + self.epsilon)
            var = (posterior[:, :, None] * (X[None, :, :] - self.mean[:, None, :]) ** 2).sum(axis=1) / \
                (posterior.sum(axis=1)[:, None] + self.epsilon)
            self.std = np.sqrt(var)
            self.prior = posterior.sum(axis=1)
            self.prior /= (self.prior.sum() + self.epsilon)

    def predict(self, X):
        """return the probability of each x belonging to each gaussian distribution"""
        dis = X[None, :, :] - self.mean[:, None, :]
        # likelihook is of shape [k, n, feature_size]
        log_likelihood = -dis ** 2 / 2 / (self.std[:, None, :] ** 2 + self.epsilon) \
            - np.log(np.sqrt(2 * np.pi) + self.epsilon) - np.log(self.std[:, None, :] + self.epsilon)
        log_likelihood = log_likelihood.sum(-1)
        # reduce likelihood to shape [k, n]
        likelihood = np.exp(log_likelihood)
        # the posterior of each datium belonging to a distribution, of shape [k, n]
        posterior = self.prior[:, None] * likelihood
        posterior /= (posterior.sum(axis=0, keepdims=True) + self.epsilon)
        return posterior


if __name__ == '__main__':
    def demonstrate(desc, X):
        gmm = GMM(3)
        gmm.fit(X)
        pred = gmm.predict(X).T
        plt.scatter(X[:, 0], X[:, 1], color=pred)
        plt.title(desc)
        plt.show()

    # ---------------------- Eample 1---------------------------------------------
    X = np.concatenate([
        np.random.normal([0, 0], [.3, .3], [100, 2]),
        np.random.normal([0, 1], [.3, .3], [100, 2]),
        np.random.normal([1, 0], [.3, .3], [100, 2]),
    ])
    demonstrate("Example 1", X)

    # ---------------------- Eample 2---------------------------------------------
    X = np.concatenate([
        np.random.normal([0, 0], [.4, .4], [100, 2]),
        np.random.normal([0, 1], [.4, .4], [100, 2]),
        np.random.normal([1, 0], [.4, .4], [100, 2]),
    ])
    demonstrate("Example 2", X)

    # ---------------------- Eample 3---------------------------------------------
    X = np.concatenate([
        np.random.normal([0, 0], [.4, .4], [100, 2]),
        np.random.normal([0, 3], [.4, .4], [100, 2]),
        np.random.normal([3, 0], [.4, .4], [100, 2]),
    ])
    demonstrate("Example 3", X)

    # ---------------------- Eample 4---------------------------------------------
    X = np.concatenate([
        np.random.normal([0, 0], [.4, .4], [1, 2]),
        np.random.normal([0, 3], [.4, .4], [1, 2]),
        np.random.normal([3, 0], [.4, .4], [1, 2]),
    ])
    demonstrate("Example 4", X)
