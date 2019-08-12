
"""
This module is a set of useful function, while famous library does not seem to contain.
"""
def logcosh(x:np.ndarray):
    """
    Calculating a log cosh(x).
    When absolute value of x is very large, this function are overflow,
    so we avoid it.
    """
    return np.abs(x) + np.log((1 + np.exp(-2 * np.abs(x)))/2)

def logpdf_mixture_dist(x:np.ndarray, param:dict, component_log_dist:Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]):
    """
    対数尤度の計算
    確率分布が混合分布の時を想定:
    $\log p(x|w) = \sum_{i = 1}^n \log p(x_i|w) = \sum_{i = 1}^n \log \exp(L_{ik}(w)) =  \sum_{i = 1}^n \{\hat{L}_{i} + \log \exp(L_{ik}(w) - \hat{L}(i)) \}$,
    where L_{ik} = \log a_k + \log p(x_i|w, y_{ik} = 1)\hat{L}(i) = \max_{k} L_{ik}, p(x_i|w, y_{ik}=1):i番目のサンプルのk番目のクラスタの確率分布

    + 入力:
        1. x:入力データ(n*M)
        2. param: 確率分布のパラメータ(ratio: 混合比, mean: 各クラスタの平均値 K*M, scale: 各クラスタのscale(正規分布における標準偏差) K*M)
        3. component_log_dist: 各クラスタの対数確率密度の値 logp(x|w,y)
    """
    n = x.shape[0]
    K = len(param["ratio"])
    loglik = np.zeros((n,K))
    for k in range(K):
        if param["scale"].ndim == 2:
            loglik[:,k] = np.log(param["ratio"][k]) + component_log_dist(test_x, param["mean"][k,:],  param["scale"][k,:])
        elif param["scale"].ndim == 3:
            loglik[:,k] = np.log(param["ratio"][k]) + component_log_dist(test_x, param["mean"][k,:],  param["scale"][k,:,:])
        else:
            raise ValueError("Error precision, dimension of precision must be 2 or 3!")
    max_loglik = loglik.max(axis = 1)
    norm_loglik = loglik - np.repeat(max_loglik,K).reshape(n,K)
    return (np.log(np.exp(norm_loglik).sum(axis = 1)) + max_loglik)
