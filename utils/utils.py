from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    from scipy import sparse
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.preprocessing import MaxAbsScaler, StandardScaler
    import numpy as np


def preprocess_data(X, y=None, remove_zerovar=True, standardize=True):
    if remove_zerovar:
        X = VarianceThreshold().fit_transform(X)

    if standardize:
        if sparse.issparse(X):
            X = MaxAbsScaler().fit_transform(X).tocsc()
        else:
            X = StandardScaler().fit_transform(X)

    return X, y


def prox_slope(beta, lambdas):
    """Compute the sorted L1 proximal operator.
    Parameters
    ----------
    beta : array
        vector of coefficients
    lambdas : array
        vector of regularization weights
    Returns
    -------
    array
        the result of the proximal operator
    """
    beta_sign = np.sign(beta)
    beta = np.abs(beta)
    ord = np.flip(np.argsort(beta))
    beta = beta[ord]

    p = len(beta)

    s = np.empty(p, np.float64)
    w = np.empty(p, np.float64)
    idx_i = np.empty(p, np.int64)
    idx_j = np.empty(p, np.int64)

    k = 0

    for i in range(p):
        idx_i[k] = i
        idx_j[k] = i
        s[k] = beta[i] - lambdas[i]
        w[k] = s[k]

        while (k > 0) and (w[k - 1] <= w[k]):
            k = k - 1
            idx_j[k] = i
            s[k] += s[k + 1]
            w[k] = s[k] / (i - idx_i[k] + 1)

        k = k + 1

    for j in range(k):
        d = max(w[j], 0.0)
        for i in range(idx_i[j], idx_j[j] + 1):
            beta[i] = d

    beta[ord] = beta.copy()
    beta *= beta_sign

    return beta
