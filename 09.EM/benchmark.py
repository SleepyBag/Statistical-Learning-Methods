from tabnanny import verbose
import numpy as np
from matplotlib import pyplot as plt

from GMM import GMM
from GMMGradientDescent import GMMGradientDescent

def compare(X, k):
    gmm = GMM(k, verbose=False)
    gmm_gradient_descent = GMMGradientDescent(k, verbose=False)
    gmm.fit(X)
    gmm_gradient_descent.fit(X)
    gmm_likelihood = np.exp(gmm.log_likelihood(X))
    gmm_gradient_descent_log_likelihood = np.exp(gmm_gradient_descent.log_likelihood(X))
    return gmm_likelihood, gmm_gradient_descent_log_likelihood

X = np.concatenate([
    np.random.normal([0, 0], [.3, .3], [100, 2]),
    np.random.normal([0, 1], [.3, .3], [100, 2]),
    np.random.normal([1, 0], [.3, .3], [100, 2]),
])
gmm_likelihoods = []
gmm_gradient_descent_likelihoods = []
for i in range(50):
    print('Running comparison', i)
    gmm_likelihood, gmm_gradient_descent_likelihood = compare(X, 3)
    gmm_likelihoods.append(gmm_likelihood)
    gmm_gradient_descent_likelihoods.append(gmm_gradient_descent_likelihood)
    print('likelihood of EM algorithm is', gmm_likelihood)
    print('likelihood of gradient descent is', gmm_gradient_descent_likelihood)

plt.boxplot([gmm_likelihoods, gmm_gradient_descent_likelihoods])
# plt.axes().set_xticklabels(["EM", "gradient descent"])
plt.show()