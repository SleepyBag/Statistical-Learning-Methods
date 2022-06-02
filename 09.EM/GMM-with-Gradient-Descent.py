import torch
import math
from matplotlib import pyplot as plt

class GMM:
    def __init__(self, k, independent_variance=True, max_step=20000, learning_rate=1e-3, verbose=True):
        self.k = k
        self.max_step = max_step
        self.epsilon = 1e-8
        self.learning_rate = learning_rate
        self.log_sqrt_2pi = math.log(math.sqrt(2 * torch.pi))
        self.verbose = verbose
        # specify whether each feature has independent variance - that is, has a diagnol covariance matrix
        self.independent_variance = independent_variance
        if not independent_variance:
            raise NotImplementedError("GMM with Gradient Descent is not implemented yet because of the difficulty of dealing with covariance matrix")

    def fit(self, X):
        """
        X: training data of shape [n, feature_size]
        """
        n, self.feature_size = X.shape
        X = torch.Tensor(X)
        # the parameter of each gaussian distribution
        self.prior_logit = torch.zeros(self.k)
        self.prior_logit.requires_grad_()
        if self.independent_variance:
            self.log_std = torch.log(X.std(dim=0)).repeat(self.k, 1)
            self.log_std.requires_grad_()
            self.mean = torch.zeros(self.k, self.feature_size)
            self.mean.normal_()
            self.mean.requires_grad_()
        else:
            raise NotImplementedError("GMM with Gradient Descent is not implemented yet because of the difficulty of dealing with covariance matrix")
        self.optimizer = torch.optim.Adam([self.log_std, self.mean, self.prior_logit], lr=self.learning_rate)

        for step in range(self.max_step):
            ##########################################
            # Calculate Likelihood
            ##########################################
            # posterior probability of each sample in each Gaussian model
            # it is exactly the likelihood of parameters including mean, std and prior
            likelihood = self.likelihood(X)
            neg_log_likelihood = -torch.log(likelihood.sum(dim=0)).mean()

            if self.verbose:
                if step % 1000 == 0:
                    print('Step', step, ', negative log likelihood is', neg_log_likelihood)

            ##########################################
            # Gradient Descent Step
            ##########################################
            self.optimizer.zero_grad()
            neg_log_likelihood.backward()
            self.optimizer.step()

    def likelihood(self, X):
        """return the probability of each x belonging to each gaussian distribution"""
        # dis[i, j, k] is the distance from i-th center to j-th sample, in k-th dimension
        dis = X[None, :, :] - self.mean[:, None, :]

        # calculate log likelihood first, then likelihood
        if self.independent_variance:
            # log_likelihook is of shape [k, n, feature_size]
            log_likelihood = -dis ** 2 * .5 / (torch.exp(self.log_std[:, None, :]) ** 2 + self.epsilon) \
                - self.log_sqrt_2pi - self.log_std[:, None, :]
            # reduce likelihood to shape [k, n]
            # log_likelihood[i, j] is the likelihood of j-th sample belonging to i-th center
            log_likelihood = log_likelihood.sum(dim=-1)
        else:
            # log_likelihook is of shape [k, n]
            # log_likelihood[i, j] is the likelihood of j-th sample belonging to i-th center
            fixed_cov = self.cov + self.epsilon * torch.eye(self.feature_size)
            log_likelihood = -.5 * (dis @ torch.linalg.inv(fixed_cov) * dis).sum(axis=-1) \
                -.5 * torch.linalg.slogdet(2 * torch.pi * fixed_cov)[1][:, None]                            # slogdet returns [sign, logdet], we just need logdet

        likelihood = torch.exp(log_likelihood)
        # the posterior of each datium belonging to a distribution, of shape [k, n]
        likelihood = torch.nn.functional.softmax(self.prior_logit)[:, None] * likelihood
        return likelihood

    def predict(self, X):
        posterior = self.likelihood(torch.Tensor(X)).detach().numpy()
        posterior /= (posterior.sum(axis=0, keepdims=True) + self.epsilon)
        return posterior


if __name__ == '__main__':
    import numpy as np

    def demonstrate(desc, X):
        gmm = GMM(3, independent_variance=True)
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
