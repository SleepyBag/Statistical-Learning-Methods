import numpy as np
from matplotlib import pyplot as plt

class GMM:
    def __init__(self, k, independent_variance=True, max_step=2000, verbose=True):
        self.k = k
        self.max_step = max_step
        self.epsilon = 1e-8
        self.verbose = verbose
        # specify whether each feature has independent variance - that is, has a diagnol covariance matrix
        self.independent_variance = independent_variance

    def fit(self, X):
        """
        X: training data of shape [n, feature_size]
        """
        n, self.feature_size = X.shape
        # the parameter of each gaussian distribution
        self.prior = np.ones(self.k) / self.k
        self.prior /= self.prior.sum()
        if self.independent_variance:
            self.std = np.repeat(np.std(X, axis=0, keepdims=True), self.k, axis=0)
            self.mean = np.random.normal(X.mean(axis=0), self.std, [self.k, self.feature_size])
        else:
            self.cov = np.repeat(np.cov(X.T)[None, ...], self.k, axis=0)
            self.mean = np.random.multivariate_normal(X.mean(axis=0), self.cov[0], [self.k])

        pre_likelihood = np.zeros([self.k, n])
        for step in range(self.max_step):
            ##########################################
            # Expectation step
            ##########################################
            # posterior probability of each sample in each Gaussian model
            posterior = self.predict(X)
            if self.verbose():
                print('Step', step, ', posterior probability of data is', posterior)

            ##########################################
            # Maximization step
            ##########################################
            # center of each Gaussian model
            self.mean = (posterior[:, :, None] * X[None, :, :]).sum(axis=1) / \
                (posterior.sum(axis=1)[:, None] + self.epsilon)
            # distance from each sample to each center
            dis = X[None, :, :] - self.mean[:, None, :]
            if self.independent_variance:
                # variance of each Gaussian model
                var = (posterior[:, :, None] * dis ** 2).sum(axis=1) / \
                    (posterior.sum(axis=1)[:, None] + self.epsilon)
                # standard deviation of each Gaussian model, in each dimension
                # shape [k, feature_size]
                # std[i, j] is the variance of j-th feature in the i-th Gaussian model
                self.std = np.sqrt(var)
            else:
                # covariance of each Gaussian model
                # shape [k, feature_size, feature_size]
                # cov[i] is the covariance matrix of i-th Gaussian model
                self.cov =  (dis.transpose(0, 2, 1) @ (posterior[:, :, None] * dis)) / \
                    (posterior.sum(axis=1)[:, None, None] + self.epsilon)
            self.prior = posterior.sum(axis=1)
            self.prior /= (self.prior.sum() + self.epsilon)

            if (self.likelihood - pre_likelihood).max() < self.epsilon:
                break
            pre_likelihood = self.likelihood

    def predict(self, X):
        """return the probability of each x belonging to each gaussian distribution"""
        # dis[i, j, k] is the distance from i-th center to j-th sample, in k-th dimension
        dis = X[None, :, :] - self.mean[:, None, :]

        # calculate log likelihood first, then likelihood
        if self.independent_variance:
            # log_likelihook is of shape [k, n, feature_size]
            log_likelihood = -dis ** 2 * .5 / (self.std[:, None, :] ** 2 + self.epsilon) \
                - np.log(np.sqrt(2 * np.pi) + self.epsilon) - np.log(self.std[:, None, :] + self.epsilon)
            # reduce likelihood to shape [k, n]
            # log_likelihood[i, j] is the likelihood of j-th sample belonging to i-th center
            log_likelihood = log_likelihood.sum(-1)
        else:
            # log_likelihook is of shape [k, n]
            # log_likelihood[i, j] is the likelihood of j-th sample belonging to i-th center
            fixed_cov = self.cov + self.epsilon * np.eye(self.feature_size)
            log_likelihood = -.5 * (dis @ np.linalg.inv(fixed_cov) * dis).sum(axis=-1) \
                -.5 * np.linalg.slogdet(2 * np.pi * fixed_cov)[1][:, None]                            # slogdet returns [sign, logdet], we just need logdet

        likelihood = np.exp(log_likelihood)
        self.likelihood = likelihood
        # the posterior of each datium belonging to a distribution, of shape [k, n]
        posterior = self.prior[:, None] * likelihood
        posterior /= (posterior.sum(axis=0, keepdims=True) + self.epsilon)
        return posterior


if __name__ == '__main__':
    def demonstrate(desc, X):
        gmm = GMM(3, independent_variance=False)
        gmm.fit(X)
        pred = gmm.predict(X).T
        plt.scatter(X[:, 0], X[:, 1], color=pred)
        plt.title(desc)
        plt.show()

    # ---------------------- Example 1---------------------------------------------
    X = np.concatenate([
        np.random.normal([0, 0], [.3, .3], [100, 2]),
        np.random.normal([0, 1], [.3, .3], [100, 2]),
        np.random.normal([1, 0], [.3, .3], [100, 2]),
    ])
    demonstrate("Example 1", X)

    # ---------------------- Example 2---------------------------------------------
    demonstrate("Example 2: GMM does'nt promise the same result for the same data", X)

    # ---------------------- Example 3---------------------------------------------
    X = np.concatenate([
        np.random.normal([0, 0], [.4, .4], [100, 2]),
        np.random.normal([0, 1], [.4, .4], [100, 2]),
        np.random.normal([1, 0], [.4, .4], [100, 2]),
    ])
    demonstrate("Example 3", X)

    # ---------------------- Example 4---------------------------------------------
    X = np.concatenate([
        np.random.normal([0, 0], [.4, .4], [100, 2]),
        np.random.normal([0, 3], [.4, .4], [100, 2]),
        np.random.normal([3, 0], [.4, .4], [100, 2]),
    ])
    demonstrate("Example 4", X)

    # ---------------------- Example 5---------------------------------------------
    X = np.concatenate([
        np.random.normal([0, 0], [.4, .4], [1, 2]),
        np.random.normal([0, 3], [.4, .4], [1, 2]),
        np.random.normal([3, 0], [.4, .4], [1, 2]),
    ])
    demonstrate("Example 5", X)
