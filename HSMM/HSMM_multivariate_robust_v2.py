# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
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

# %matplotlib inline

# # Evaluating robustness of Multivariate Hyperbolic Secant Mixture model(HSMM)
# + This notebook compares HSMM with Gaussian Mixture Model (GMM) in terms of counting correct labeling.

# ## Method:
# + As the cluster distribution, the following distributions are used, and compare clustering performance:
#
#
# 1. Gaussian distribution
# 1. Hyperbolic secant distribution
# 1. Laplace distribution
# 1. Gumbel distribution
# 1. Student distribution
# 1. Cauchy distribution
# 1. Gamma distribution
# 1. Skew Normal distribution

# ### Suplementary material: Local Variational Approximation algorithm for HSMM
# + Model:
#     + $p(x|w) = \sum_{k=1}^K a_k \prod_{j = 1}^M \frac{\sqrt{s_{kj}}}{2} \frac{1}{\cosh(\frac{s_{kj}}{2}(x_j - b_{kj}))}$
#     + $x, b_k \in \mathbb{R}^M, s_k \in \mathbb{R}_+^M$
# + Prior distribution:
#     + $\varphi(w) = Dir(a|\{ \alpha_k \}_{k=1}^K) \prod_{k=1}^K N(b_k|0, (s_k \beta_k)^{-1} ) Gam(s_k|\gamma_k, \delta_k)$
# + Algorithm
#    1. Initializing the following values:
#        + $g_{ikj}(\eta), v_{ikj}(\eta), h_{ik}(\xi), u_{ik}(\xi)$
#    2. Update the following values
#        + $\hat{\alpha}_k = \alpha_k + \sum_{i=1}^n u_{ik}(\xi)$
#        + $\hat{\beta}_{kj} = \beta_k + \sum_{i=1}^n -2v_{ikj}(\eta)$
#        + $\hat{m}_{kj} = \frac{1}{\hat{\beta}_k} \sum_{i=1}^n -2v_{ik}(\eta)x_i$
#        + $\hat{\gamma}_{kj} = \gamma_k + \frac{1}{2}\sum_{i=1}^n u_{ik}(\xi)$
#        + $\hat{\delta}_{kj} = \delta_k + \sum_{i=1}^n -v_{ikj}(\eta)x_{ij}^2 - \frac{\hat{\beta}_{kj}}{2}\hat{m}_{kj}^2$
#    3. Update the following values
#        + $g_{ikj}(\eta) = \frac{\hat{\gamma}_{kj}}{\hat{\delta}_{kj}} (x_{ij} - \hat{m}_{kj})^2 + \frac{1}{\hat{\beta}_{kj}}$
#        + $v_{ikj}(\eta) = -u_{ik}(\xi)\frac{ \tanh(\sqrt{g_{ikj}(\eta)}/2) }{4\sqrt{g_{ikj}(\eta)}}$
#    4. Update the following values
#        + $h_{ik}(\xi) = \psi(\hat{\alpha}_k) - \psi(\sum_{l=1}^K \hat{\alpha}_l) + \frac{1}{2} \sum_{j=1}^M (\psi(\hat{\gamma}_{kj}) - \log(\hat{\delta}_{kj})) - \sum_{j=1}^M \log(\cosh(\sqrt{g_{ikj}(\eta)}/2)) $
#        + $u_{ik}(\xi) = \frac{ \exp(h_{ik}(\xi)) }{ \sum_{l=1}^K \exp(h_{il}(\xi)) }$
#        + ここで,$\psi(x) = \frac{d}{dx}\log \Gamma(x)$
#    5. Return back to 2.
#    
# + Evaluation function $\overline{F}_{\xi, \eta}(x^n)$:
#     + $\overline{F}_{\xi, \eta}(x^n) = - \phi(h(\xi)) - \psi(g(\eta)) + u(\xi) \cdot h(\xi) + v(\eta) \cdot g(\eta) $  
#         $+ nM \log 2 \pi + \log \Gamma(\sum_{l = 1}^K \hat{\alpha}_l) - \log \Gamma({\sum_{l = 1}^K\alpha}_l) + \sum_{k=1}^K \log \frac{\Gamma(\alpha_k)}{\Gamma(\hat{\alpha}_k)}$  
#         $+ \sum_{k=1}^K \sum_{j=1}^M \bigl\{ \frac{1}{2} \log \frac{\hat{\beta}_{kj}}{\beta_{kj}} + \hat{\gamma}_{kj} \log \hat{\delta}_{kj} - \gamma_{kj} \log \delta_{kj} - \log \Gamma(\hat{\gamma}_{kj}) + \log \Gamma(\gamma_{kj}) \bigr\}$

from IPython.core.display import display, Markdown, Latex
import math
import numpy as np
from scipy.special import gammaln, psi
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, t, cauchy, laplace, gumbel_r, gamma, skewnorm


# ## Used funtions

def printmd(x):
    display(Markdown(x))


def random_hsm(n, loc = 0, scale = 1):
    """
    Generate data following hyperbolic secant distribution.
    Let $Y \sim standard_cauchy(x)$,  
    random variable $X = \frac{2}{s}\sinh^{-1}(Y) + b$ follows to  
    $X \sim p(x) = \frac{s}{2\pi}\frac{1}{\cosh(s(x-b)/2)}$.
    """
    Y = np.random.standard_cauchy(size=n)
    X = 2/np.sqrt(scale)*np.arcsinh(Y) + loc    
    return X


def fit_lva_gmm(train_X:np.ndarray, K:int,
                 pri_alpha = 0.1, pri_beta = 0.001, pri_gamma = 2, pri_delta = 2,
                 iteration = 1000, restart_num:int = 5, learning_seeds:list = None):
    """
    LVA for GMM.
    This is same with Variational Bayes inference for GMM.
    Since the algorithm fails to local minima, the best estimator are chosen in several initial values.
    
    + Input:
        + train_X: input data
        + pri_alpha: hyperparameter for prior distribution of symmetric Dirichlet distribution.
        + pri_beta: hyperparameter for prior distribution of Normal distribution for inverse variance.
        + pri_gamma: hyperparameter for prior distribution of Gamma distribution for shape parameter.
        + pri_delta: hyperparameter for prior distribution of Gamma distribution for rate parameter.
        + iteration: Number of iteration.
        + restart_num: Number of restart of inital values.
        + learning_seeds: Seeds for initial values.
        
    + Output:
        + Dictionary of the best estimated result:
            1. alpha: parameter for posterior distribution of Dirichlet distribution.
            2. mu: parameter for posterior distribution of Normal distribution for mean parameter.
            3. beta: parameter for posterior distribution of Normal distribution for inverse variance parameter.
            4. gamma: parameter for posterior distribution of Gamma distribution for shape parameter.
            5. delta: parameter for posterior distribution of Gamma distribution for rate parameter.
            6. h_xi: Value of E_w[log p(x_i, z_i = k|w)], where z_i is latent variable. This parameters form posterior latent distribution.
            7. u_xi: Value of p(z_i = k). This parameters represent posterior probability of latent variable
            8. energy: Value of the best evaluation function.
            9. seed: Value of the best learning seed.
    """
    
    (n, M) = train_X.shape
    ### Setting for static variable in the algorithm.
    expand_x = np.repeat(x, K).reshape(n, M, K).transpose((0, 2, 1)) ### n * K * M data with the same matrix among 2nd dimension

    min_energy = np.inf
    result = dict()
    
    for restart in range(restart_num):
        ### Set learning seed if learning_seeds is specified.
        if learning_seeds is not None and len(learning_seeds) >= restart:
            np.random.seed(learning_seeds[restart])

        energy = np.zeros(iteration)
        ### Setting for initial value
        est_u_xi = np.random.dirichlet(alpha = np.ones(K), size=n)
    
        ### Start learning.
        for ite in range(iteration):
            ### Update posterior distribution of parameter.
            est_alpha = pri_alpha + est_u_xi.sum(axis = 0)
            est_beta = np.repeat(pri_beta + est_u_xi.sum(axis = 0), M).reshape(K,M)
            est_m = est_u_xi.T @ x / est_beta
            est_gamma = np.repeat(pri_gamma + est_u_xi.sum(axis = 0)/2, M).reshape(K,M)
            est_delta = pri_delta + est_u_xi.T @ (x**2) /2 - est_beta / 2 * est_m**2
            
            ### Update posterior distribution of latent variable
            est_g_eta = np.repeat(est_gamma / est_delta, n).reshape(K,M,n).transpose((2,0,1)) * (expand_x - np.repeat(est_m,n).reshape(K,M,n).transpose((2,0,1)))**2 + 1/np.repeat(est_beta, n).reshape(K,M,n).transpose((2,0,1))
            est_h_xi = -M/2*np.log(2*np.pi) + np.repeat(psi(est_alpha) - psi(est_alpha.sum()) + (psi(est_gamma) - np.log(est_delta)).sum(axis = 1)/2, n).reshape(K,n).T - est_g_eta.sum(axis = 2)/2
            max_h_xi = est_h_xi.max(axis = 1)
            norm_h_xi = est_h_xi - np.repeat(max_h_xi,K).reshape(n,K)
            est_u_xi = np.exp(norm_h_xi) / np.repeat(np.exp(norm_h_xi).sum(axis = 1), K).reshape(n,K)

            ### Calculate evaluation function
            energy[ite] = - (np.log(np.exp(norm_h_xi).sum(axis = 1)) + max_h_xi).sum() + (est_u_xi * est_h_xi).sum()
            energy[ite] += gammaln(est_alpha.sum()) - gammaln(K*pri_alpha) + (-gammaln(est_alpha) + gammaln(pri_alpha)).sum()
            energy[ite] += (np.log(est_beta/pri_beta)/2 + est_gamma * np.log(est_delta) - pri_gamma * np.log(pri_delta) - gammaln(est_gamma) + gammaln(pri_gamma)).sum()
        
        print(energy[-1])        
        if energy[-1] < min_energy:
            min_energy = energy[-1]
            result["alpha"] = est_alpha
            result["mu"] = est_m
            result["beta"] = est_beta
            result["gamma"] = est_gamma
            result["delta"] = est_delta
            result["h_xi"] = est_h_xi
            result["u_xi"] = est_u_xi
            result["energy"] = energy
            if learning_seeds is not None and len(learning_seeds) >= restart:
                result["seed"] = learning_seeds[restart]
    return result


def fit_lva_hsmm(train_X:np.ndarray, K:int,
                 pri_alpha = 0.1, pri_beta = 0.001, pri_gamma = 2, pri_delta = 2,
                 iteration = 1000, restart_num:int = 5, learning_seeds:list = None):
    """
    LVA for HSMM.
    The algorithm is described in the above cell.
    Since the algorithm fails to local minima, the best estimator are chosen in several initial values.
    
    + Input:
        + train_X: input data
        + pri_alpha: hyperparameter for prior distribution of symmetric Dirichlet distribution.
        + pri_beta: hyperparameter for prior distribution of Normal distribution for inverse variance.
        + pri_gamma: hyperparameter for prior distribution of Gamma distribution for shape parameter.
        + pri_delta: hyperparameter for prior distribution of Gamma distribution for rate parameter.
        + iteration: Number of iteration.
        + restart_num: Number of restart of inital values.
        + learning_seeds: Seeds for initial values.
        
    + Output:
        + Dictionary of the best estimated result:
            1. alpha: parameter for posterior distribution of Dirichlet distribution.
            2. mu: parameter for posterior distribution of Normal distribution for mean parameter.
            3. beta: parameter for posterior distribution of Normal distribution for inverse variance parameter.
            4. gamma: parameter for posterior distribution of Gamma distribution for shape parameter.
            5. delta: parameter for posterior distribution of Gamma distribution for rate parameter.
            6. h_xi: Value of E_w[log p(x_i, z_i = k|w)], where z_i is latent variable. This parameters form posterior latent distribution.
            7. u_xi: Value of p(z_i = k). This parameters represent posterior probability of latent variable.
            8. g_eta: Value of auxiliary variable, which represents g(\eta) in the algorithm.
            9. v_eta: Value of auxiliary variable, which represents v(\eta) in the algorithm.
            10. energy: Value of the best evaluation function.
            11. seed: Value of the best learning seed.
    """    
    
    (n, M) = train_X.shape
    ### Setting for static variable in the algorithm.
    expand_x = np.repeat(x, K).reshape(n, M, K).transpose((0, 2, 1)) ### n * K * M data with the same matrix among 2nd dimension

    min_energy = np.inf
    result = dict()
    
    for restart in range(restart_num):
        ### Set learning seed if learning_seeds is specified.
        if learning_seeds is not None and len(learning_seeds) >= restart:
            np.random.seed(learning_seeds[restart])

        energy = np.zeros(iteration)
        ### Setting for initial value
        est_u_xi = np.random.dirichlet(alpha = np.ones(K), size=n)

        est_g_eta = np.abs(np.random.normal(size=(n,K,M)))
        est_v_eta = - np.repeat(est_u_xi, M).reshape(n, K, M) * np.tanh(np.sqrt(est_g_eta)/2)/(4*np.sqrt(est_g_eta))

        ### Start learning.
        for ite in range(iteration):
            ### Update posterior distribution of parameter
            est_alpha = pri_alpha + est_u_xi.sum(axis = 0)
            est_beta = pri_beta + (-2*est_v_eta.sum(axis = 0))
            est_m = -2 * (expand_x * est_v_eta).sum(axis = 0) / est_beta
            est_gamma = np.repeat(pri_gamma + est_u_xi.sum(axis = 0)/2, M).reshape(K,M)
            est_delta = pri_delta - (expand_x**2 * est_v_eta).sum(axis = 0) - est_beta / 2 * est_m**2
            
            ### Update auxiliary variables
            est_g_eta = np.repeat(est_gamma / est_delta, n).reshape(K,M,n).transpose((2,0,1)) * (expand_x - np.repeat(est_m,n).reshape(K,M,n).transpose((2,0,1)))**2 + 1/np.repeat(est_beta, n).reshape(K,M,n).transpose((2,0,1))
            est_v_eta = - np.repeat(est_u_xi, M).reshape(n, K, M) * np.tanh(np.sqrt(est_g_eta)/2)/(4*np.sqrt(est_g_eta))

            ### Update posterior distribution of latent variable
            sqrt_g_eta = np.sqrt(est_g_eta)
            est_h_xi = np.repeat(psi(est_alpha) - psi(est_alpha.sum()) + (psi(est_gamma) - np.log(est_delta)).sum(axis = 1)/2, n).reshape(K,n).T - (sqrt_g_eta/2 +  np.log( (1 + np.exp(-2*sqrt_g_eta/2 ))/2)).sum(axis = 2)
            max_h_xi = est_h_xi.max(axis = 1)
            norm_h_xi = est_h_xi - np.repeat(max_h_xi,K).reshape(n,K)
            est_u_xi = np.exp(norm_h_xi) / np.repeat(np.exp(norm_h_xi).sum(axis = 1), K).reshape(n,K)

            ### Calculate evaluation function
            energy[ite] = (np.repeat(est_u_xi, M).reshape(n, K, M) * (sqrt_g_eta/2 +  np.log( (1 + np.exp(-2*sqrt_g_eta/2 ))/2)) ).sum() - (np.log(np.exp(norm_h_xi).sum(axis = 1)) + max_h_xi).sum() + (est_u_xi * est_h_xi).sum() + (est_v_eta * est_g_eta).sum()
            energy[ite] += gammaln(est_alpha.sum()) - gammaln(K*pri_alpha) + (-gammaln(est_alpha) + gammaln(pri_alpha)).sum()
            energy[ite] += (np.log(est_beta/pri_beta)/2 + est_gamma * np.log(est_delta) - pri_gamma * np.log(pri_delta) - gammaln(est_gamma) + gammaln(pri_gamma)).sum()
        
        print(energy[-1])        
        if energy[-1] < min_energy:
            min_energy = energy[-1]
            result["alpha"] = est_alpha
            result["beta"] = est_beta
            result["mu"] = est_m
            result["gamma"] = est_gamma
            result["delta"] = est_delta
            result["h_xi"] = est_h_xi
            result["u_xi"] = est_u_xi
            result["g_eta"] = est_g_eta
            result["v_eta"] = est_v_eta            
            result["energy"] = energy
            if learning_seeds is not None and len(learning_seeds) >= restart:
                result["seed"] = learning_seeds[restart]
    return result


import itertools
def evaluate_correct_cluster_number(result:dict, noise_data_num:int, true_label_arg, K:int):    
    est_label_prob = result["u_xi"]
    if noise_data_num > 0:
        target_label_arg = true_label_arg[:-noise_data_num]
        est_label_arg = np.argmax(est_label_prob, axis = 1)[:-noise_data_num]
    else:
        target_label_arg = true_label_arg
        est_label_arg = np.argmax(est_label_prob, axis = 1)

    max_correct_num = 0
    for perm in list(itertools.permutations(range(K), K)):
        permed_est_label_arg = est_label_arg.copy()
        for i in range(len(perm)):
            permed_est_label_arg[est_label_arg == i] = perm[i]
        correct_num = (permed_est_label_arg == target_label_arg).sum()
        if correct_num > max_correct_num:
            max_correct_num = correct_num
            max_perm = perm
            max_est_label_arg = permed_est_label_arg
    return (max_correct_num, max_perm, max_est_label_arg)


def learning_and_labeling():
    printmd("### 1. Data distribution:")
    plot_scatter_with_label(x, true_label_arg,  K0, noise_data_num)

    printmd("### 2. Learning by GMM:")
    gmm_result = fit_lva_gmm(x, K, pri_alpha = pri_alpha, pri_beta = pri_beta, pri_gamma = pri_gamma, pri_delta = pri_delta, learning_seeds = learning_seeds)
    print("mean plug-in parameters: \n {0}".format({
        "est_ratio": gmm_result["alpha"] / sum(gmm_result["alpha"]),
        "est_mean": gmm_result["mu"],
        "est_precision": gmm_result["gamma"] / gmm_result["delta"]
    }))
    (correct_num_gmm, perm_gmm, label_arg_gmm) = evaluate_correct_cluster_number(gmm_result, noise_data_num, true_label_arg, K)

    printmd("### 3.. Learning by HSMM:")
    hsmm_result = fit_lva_hsmm(x, K, pri_alpha = pri_alpha, pri_beta = pri_beta, pri_gamma = pri_gamma, pri_delta = pri_delta, learning_seeds=learning_seeds)
    print("mean plug-in parameters: \n {0}".format({
        "est_ratio": hsmm_result["alpha"] / sum(hsmm_result["alpha"]),
        "est_mean": hsmm_result["mu"],
        "est_precision": hsmm_result["gamma"] / hsmm_result["delta"]
    }))
    (correct_num_hsmm, perm_hsmm, label_arg_hsmm) = evaluate_correct_cluster_number(hsmm_result, noise_data_num, true_label_arg, K)

    printmd("### 4. Correct number of labeling of GMM:")
    printmd("+ {0}/{1}".format(correct_num_gmm, len(label_arg_hsmm)))

    printmd("### 5. Correct number of labeling of HSMM:")
    printmd("+ {0}/{1}".format(correct_num_hsmm, len(label_arg_hsmm)))

    printmd("### 6. Data distribution labeled by GMM:")
    plot_scatter_with_label(x, label_arg_gmm,  K, noise_data_num)

    printmd("### 7. Data distribution labeled by HSMM:")
    plot_scatter_with_label(x, label_arg_hsmm,  K, noise_data_num)


def plot_scatter_with_label(x:np.ndarray, label_arg:np.ndarray,  K:int, noise_data_num):
    """
    Scatter plot for data x
    """
    for i in range(K):
        if noise_data_num > 0:
            plt.scatter(x[np.where(label_arg[:-noise_data_num] == i)[0],0], x[np.where(label_arg[:-noise_data_num] == i)[0],1])
        else:
            plt.scatter(x[np.where(label_arg == i)[0],0], x[np.where(label_arg == i)[0],1])        
    plt.show()


# ## Problem setting:

data_seed = 20190522
true_ratio = np.array([0.33, 0.33, 0.34])
true_delta = 0
true_s = np.array([[2, 2], [0.5, 0.5], [1, 1]])
true_b = np.array([[4, 4], [-4, -4], [0, 0]])
n = 2000
M = true_b.shape[1]
K0 = len(true_ratio)
np.random.seed(data_seed)

# ## Learning setting:

# +
### Iteration settings
iteration = 1000
learning_seeds = [20190511, 20190512, 20190513, 20190514, 20190515]

## Number of components for cluster
K = 5

## Hyperparameters
pri_alpha = 0.1
pri_beta = 0.001
pri_gamma = 2
pri_delta = 2


# -

# ### Label setting for each data
# + Remark: Label is fixed through each cluster distribution.

true_label = np.random.multinomial(n = 1, pvals = true_ratio, size = n)
true_label_arg = np.argmax(true_label, axis = 1)

# ## 1. Cluster distribution is Gaussian distribution

# +
np.random.seed(data_seed)
x = np.zeros((n, M))
for i in range(n):
    for j in range(M):
        x[i, j] = norm.rvs(loc=true_b[true_label_arg[i],j], scale=true_s[true_label_arg[i],j], size = 1)

noise_data_num = math.ceil(n*true_delta)
if noise_data_num > 0:
    x[-noise_data_num:,:] = np.random.uniform(low=-30, high=30, size = noise_data_num*M).reshape(noise_data_num,M)
# -

learning_and_labeling()

# +
printmd("### 1. Data distribution:")
plot_scatter_with_label(x, true_label_arg,  K0, noise_data_num)

printmd("### 2. Learning by GMM:")
gmm_result = fit_lva_gmm(x, K, pri_alpha = pri_alpha, pri_beta = pri_beta, pri_gamma = pri_gamma, pri_delta = pri_delta, learning_seeds = learning_seeds)
print("mean plug-in parameters: \n {0}".format({
    "est_ratio": gmm_result["alpha"] / sum(gmm_result["alpha"]),
    "est_mean": gmm_result["mu"],
    "est_precision": gmm_result["gamma"] / gmm_result["delta"]
}))
(correct_num_gmm, perm_gmm, label_arg_gmm) = evaluate_correct_cluster_number(gmm_result, noise_data_num, true_label_arg, K)

printmd("### 3.. Learning by HSMM:")
hsmm_result = fit_lva_hsmm(x, K, pri_alpha = pri_alpha, pri_beta = pri_beta, pri_gamma = pri_gamma, pri_delta = pri_delta, learning_seeds=learning_seeds)
print("mean plug-in parameters: \n {0}".format({
    "est_ratio": hsmm_result["alpha"] / sum(hsmm_result["alpha"]),
    "est_mean": hsmm_result["mu"],
    "est_precision": hsmm_result["gamma"] / hsmm_result["delta"]
}))
(correct_num_hsmm, perm_hsmm, label_arg_hsmm) = evaluate_correct_cluster_number(hsmm_result, noise_data_num, true_label_arg, K)

printmd("### 4. Correct number of labeling of GMM:")
printmd("+ {0}/{1}".format(correct_num_gmm, len(label_arg_hsmm)))

printmd("### 5. Correct number of labeling of HSMM:")
printmd("+ {0}/{1}".format(correct_num_hsmm, len(label_arg_hsmm)))

printmd("### 6. Data distribution labeled by GMM:")
plot_scatter_with_label(x, label_arg_gmm,  K, noise_data_num)

printmd("### 7. Data distribution labeled by HSMM:")
plot_scatter_with_label(x, label_arg_hsmm,  K, noise_data_num)
# -

# ## 2. Cluster distribution is Hyperbolic secant distribution

# +
np.random.seed(data_seed)
x = np.zeros((n, M))
for i in range(n):
    for j in range(M):
        x[i, j] = random_hsm(n=1, loc=true_b[true_label_arg[i],j], scale=true_s[true_label_arg[i],j])

noise_data_num = math.ceil(n*true_delta)
if noise_data_num > 0:
    x[-noise_data_num:,:] = np.random.uniform(low=-30, high=30, size = noise_data_num*M).reshape(noise_data_num,M)
# -

learning_and_labeling()

# ## 3. Cluster distribution is Laplace distribution

# +
np.random.seed(data_seed)
x = np.zeros((n, M))
for i in range(n):
    for j in range(M):
        x[i, j] = laplace.rvs(loc=true_b[true_label_arg[i],j], scale=1/true_s[true_label_arg[i],j], size = 1)

noise_data_num = math.ceil(n*true_delta)
if noise_data_num > 0:
    x[-noise_data_num:,:] = np.random.uniform(low=-30, high=30, size = noise_data_num*M).reshape(noise_data_num,M)
# -

learning_and_labeling()

# ## 4. Cluster distribution is Gumbel distribution

# +
np.random.seed(data_seed)
x = np.zeros((n, M))
for i in range(n):
    for j in range(M):
        x[i, j] = gumbel_r.rvs(loc=true_b[true_label_arg[i],j], scale=1/true_s[true_label_arg[i],j], size = 1)

noise_data_num = math.ceil(n*true_delta)
if noise_data_num > 0:
    x[-noise_data_num:,:] = np.random.uniform(low=-30, high=30, size = noise_data_num*M).reshape(noise_data_num,M)
# -

learning_and_labeling()

# ## 5. Cluster distribution is student distribution

# +
np.random.seed(data_seed)
x = np.zeros((n, M))
for i in range(n):
    for j in range(M):
        x[i, j] = t.rvs(df = 1.5, loc=true_b[true_label_arg[i],j], scale=1/true_s[true_label_arg[i],j], size = 1)

noise_data_num = math.ceil(n*true_delta)
if noise_data_num > 0:
    x[-noise_data_num:,:] = np.random.uniform(low=-30, high=30, size = noise_data_num*M).reshape(noise_data_num,M)
# -

learning_and_labeling()

# ## 6. Cluster distribution is Cauchy distribution

# +
np.random.seed(data_seed)
x = np.zeros((n, M))
for i in range(n):
    for j in range(M):
        x[i, j] = cauchy.rvs(loc=true_b[true_label_arg[i],j], scale=1/true_s[true_label_arg[i],j], size = 1)

noise_data_num = math.ceil(n*true_delta)
if noise_data_num > 0:
    x[-noise_data_num:,:] = np.random.uniform(low=-30, high=30, size = noise_data_num*M).reshape(noise_data_num,M)
# -

learning_and_labeling()

# ## 7. Cluster distribution is Gamma distribution
# + Remark: Actually support of gamma distribution is not whole real line, but scipy can generate data with loc on real value.

# +
np.random.seed(data_seed)
x = np.zeros((n, M))
for i in range(n):
    for j in range(M):
        x[i, j] = gamma.rvs(a = 1, loc=true_b[true_label_arg[i],j], scale=1/true_s[true_label_arg[i],j], size = 1)

noise_data_num = math.ceil(n*true_delta)
if noise_data_num > 0:
    x[-noise_data_num:,:] = np.random.uniform(low=-30, high=30, size = noise_data_num*M).reshape(noise_data_num,M)
# -

learning_and_labeling()

# ## 8. Cluster distribution is Skew Normal distribution

# +
np.random.seed(data_seed)
x = np.zeros((n, M))
for i in range(n):
    for j in range(M):
        x[i, j] = skewnorm.rvs(a = 2, loc=true_b[true_label_arg[i],j], scale=1/true_s[true_label_arg[i],j], size = 1)

noise_data_num = math.ceil(n*true_delta)
if noise_data_num > 0:
    x[-noise_data_num:,:] = np.random.uniform(low=-30, high=30, size = noise_data_num*M).reshape(noise_data_num,M)
# -

learning_and_labeling()