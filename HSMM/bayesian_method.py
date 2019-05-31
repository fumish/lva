# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Clustering by Bayesian technique
# + 2 Bayesian methods can cluster data:
#     1. Sampling the latent variables from joint distribution of parameter and latent variables.
#     2. Sampling the latetn variables from marginalized distribution over parameter.
#
# ## About method 1:
# + Procedure of method 1 is as follows:
#     1. Decide initial state of latent variables
#     2. Sampling parameter $w$ from the following distribution:
#         + $p(w|y^n) = Dir(a|\{\hat{\alpha}_k\}_{k=1}^K) \prod_{k=1}^K \prod_{j=1}^M N(b_{kj}|\hat{\mu}_{kj}, (s_{kj} \hat{\beta}_{kj})^{-1} ) Gam(s_{kj}|\hat{\gamma}_{kj}, \hat{\delta}_{kj})$, where  
#             + $\hat{\alpha}_k = \sum_{i=1}^n y_k^{(i)} + \alpha$,  
#             + $\hat{\beta}_k = \sum_{i=1}^n y_k^{(i)} + \beta$,  
#             + $\hat{\mu}_k = \frac{1}{\hat{\beta}_k} \sum_{i=1}^n y_k^{(i)} x^{(i)}$
#             + $\hat{\gamma}_{kj} = \gamma_k + \frac{1}{2}\sum_{i=1}^n y_k^{(i)}$
#             + $\hat{\delta}_{kj} = \delta_k + \frac{1}{2}\sum_{i=1}^n y_{k}^{(i)}(x_{j}^{(i)})^2 - \frac{\hat{\beta}_{kj}}{2}\hat{m}_{kj}^2$
#     3. Sampling latent variables from the following distribution:
#         + Let $L_{ik} = \log a_k + \frac{1}{2}\sum_{j=1}^M \log s_{kj} - \sum_{j=1}^M \frac{s_{kj}}{2} (x_j^{(i)} - b_{kj})^2$,
#         + Then for each $i = 1,...,n$, $p(y_k^{(i)}=1|w) = \frac{\exp(L_{ik})}{\sum_{l=1}^K \exp(L_{il})}$.
#     4. Back to 2.
#
# ## About method 2:

# %matplotlib inline

from IPython.core.display import display, Markdown, Latex
import numpy as np
from scipy.special import gammaln, psi
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import t, cauchy, laplace, gumbel_r, norm


def printmd(x):
    display(Markdown(x))


# ## Generate training data from true distribution

data_seed = 20190522
true_ratio = np.array([0.33, 0.33, 0.34])
true_delta = 0
true_s = np.array([[2, 2], [0.5, 0.5], [1, 1]])
true_b = np.array([[4, 4], [-4, -4], [0, 0]])
n = 2000
M = true_b.shape[1]
np.random.seed(data_seed)

true_label = np.random.multinomial(n = 1, pvals = true_ratio, size = n)
true_label_arg = np.argmax(true_label, axis = 1)

### true distribution is GMM
import math
x = np.zeros((n, M))
for i in range(n):
    for j in range(M):
        x[i, j] = norm.rvs(loc=true_b[true_label_arg[i],j], scale=1/true_s[true_label_arg[i],j], size = 1)
noise_data_num = math.ceil(n*true_delta)
if noise_data_num > 0:
    x[-noise_data_num:,:] = np.random.uniform(low=-30, high=30, size = noise_data_num*M).reshape(noise_data_num,M)

plt.scatter(x[:,0], x[:,1])
plt.show()

# +
### learning setting
learning_seed = 20190529
burn_in = 20
sampling_interval = 20
sampling_num = 10000

K = 3
pri_alpha = 2
pri_beta = 0.01
pri_gamma = 2
pri_delta = 2

# -

np.random.seed(learning_seed)

### 初期値を決める
latent_label = np.random.multinomial(1, pvals = np.ones(K)/K, size=n)
latent_arg = np.argmax(latent_label, axis = 1)

# +
### Calculate the steps until burn_in times
for i_burnin in range(burn_in):
    ### Samnpling from parameters
    n_k = latent_label.sum(axis = 0)
    post_alpha = n_k + pri_alpha
    post_beta = n_k + pri_beta
    post_mu = latent_label.T @ x / (np.repeat(post_beta, M).reshape(K,M))
    post_gamma = n_k/2 + pri_gamma
    post_delta = latent_label.T @ x**2/2 - np.repeat(post_beta,M).reshape(K,M) * post_mu**2 + pri_delta

    
    ### Sampling from latent variables


# -

np.random.gamma()

for ite in range(sampling_interval * sampling_num):


