# Implements a statistical test to compare the Linkage Model to the Admixture Model

import numpy as np
from scipy.stats import chi2
import Grid_Search_Linkage_diploid as grid_l


# ---------- shape helpers ----------
def _normalize(q):
    q = np.asarray(q, dtype=float)
    s = q.sum()
    if s <= 0:
        raise ValueError("q must sum to > 0")
    return q / s


def _ensure_KM(p, K, M):
    """
    Return p with shape (K,M). Accepts (K,M) or (M,K).
    """
    p = np.asarray(p, dtype=float)
    if p.shape == (K, M):
        return p
    if p.shape == (M, K):
        return p.T
    raise ValueError(f"p must have shape (K,M)=({K},{M}) or (M,K)=({M},{K}), got {p.shape}")


def _ensure_1d_X(X, M=None):
    X = np.asarray(X)
    if X.ndim == 2 and X.shape[0] == 1:
        X = X[0]
    X = X.astype(int)
    if M is not None and X.shape[0] != M:
        raise ValueError(f"X must have length M={M}, got {X.shape[0]}")
    return X


# ---------- Admixture model ----------
def loglik_AM(X, q, p):
    """
    Admixture log-likelihood for binary X:
      theta_m = sum_k q_k p[k,m]
      logL = sum_m [ x_m log(theta_m) + (1-x_m) log(1-theta_m) ]
    """
    X = _ensure_1d_X(X)
    M = len(X)
    q = _normalize(q)
    K = len(q)
    p = _ensure_KM(p, K, M)

    theta = q @ p  # shape (M,)
    eps = 1e-12
    theta = np.clip(theta, eps, 1 - eps)

    return float(np.sum(X * np.log(theta) + (1 - X) * np.log(1 - theta)))


def loglik_AM_diploid(X, q, p):
    """Admixture log-likelihood for diploid genotypes X in {0,1,2}.

    Under the diploid admixture model, the two alleles are independent with
    marginal success probability theta_m = sum_k q_k p[k,m]. Hence
      X_m ~ Binomial(n=2, theta_m).

    We include the log binomial coefficient log(2 choose X_m).
    """
    X = _ensure_1d_X(X)
    if np.any((X < 0) | (X > 2)):
        raise ValueError("Diploid genotype X must take values in {0,1,2}.")

    M = len(X)
    q = _normalize(q)
    K = len(q)
    p = _ensure_KM(p, K, M)

    theta = q @ p  # (M,)
    eps = 1e-12
    theta = np.clip(theta, eps, 1 - eps)

    # log(2 choose x): [0, log 2, 0]
    log_coef = np.zeros(M, dtype=float)
    log_coef[X == 1] = np.log(2.0)

    return float(np.sum(log_coef + X * np.log(theta) + (2 - X) * np.log(1 - theta)))


def mle_q_admixture_EM(X, p, K, q_init=None, max_iter=500, tol=1e-10):
    """
    EM for admixture proportions q with known p, binary X.

    E-step: w_{m,k} ∝ q_k * Bernoulli(x_m; p_{k,m})
    M-step: q_k = (1/M) sum_m w_{m,k}
    """
    X = _ensure_1d_X(X)
    M = len(X)
    p = _ensure_KM(p, K, M)

    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)

    if q_init is None:
        q = np.ones(K) / K
    else:
        q = _normalize(q_init)

    prev = -np.inf
    for _ in range(max_iter):
        # log Bernoulli probs for each k,m
        # log P(X_m | Z=k) = x log p + (1-x) log(1-p)
        log_em = X[None, :] * np.log(p) + (1 - X)[None, :] * np.log(1 - p)  # (K,M)

        # responsibilities in log-space
        log_w = np.log(q + eps)[:, None] + log_em
        # normalize over k
        log_w -= logsumexp(log_w, axis=0, keepdims=True)
        w = np.exp(log_w)  # (K,M)

        q_new = w.mean(axis=1)
        q_new = _normalize(q_new)

        ll = loglik_AM(X, q_new, p)
        if abs(ll - prev) < tol:
            q = q_new
            break
        prev = ll
        q = q_new

    return q


def mle_q_admixture_EM_diploid(X, p, K, q_init=None, max_iter=500, tol=1e-10):
    """EM for admixture proportions q with known p, diploid genotypes X in {0,1,2}.

    Latent variables per marker m:
      (Z_m^(a), Z_m^(b)) iid ~ q over {0,...,K-1}.
    Emission (given i,j): genotype probability under Hardy–Weinberg
      P(X=0)= (1-p_i)(1-p_j),
      P(X=1)= p_i(1-p_j) + (1-p_i)p_j,
      P(X=2)= p_i p_j.

    E-step: w_{m,i,j} ∝ q_i q_j P(X_m | i,j)
    M-step: q_k = (1/(2M)) sum_m E[ 1{Z_m^(a)=k} + 1{Z_m^(b)=k} | X_m ]

    Complexity: O(M*K^2) per iteration.
    """
    X = _ensure_1d_X(X)
    if np.any((X < 0) | (X > 2)):
        raise ValueError("Diploid genotype X must take values in {0,1,2}.")

    M = len(X)
    p = _ensure_KM(p, K, M)

    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)

    if q_init is None:
        q = np.ones(K) / K
    else:
        q = _normalize(q_init)

    # Efficient EM without K^2 loops:
    # For each marker, compute the posterior expected ancestry counts across the two alleles.
    # Let theta_m = sum_k q_k p_{k,m}. Then
    #   X=2: E[count_k] = 2 * (q_k p_{k,m}) / theta_m
    #   X=0: E[count_k] = 2 * (q_k (1-p_{k,m})) / (1-theta_m)
    #   X=1: E[count_k] = (q_k p_{k,m})/theta_m + (q_k(1-p_{k,m}))/(1-theta_m)

    prev = -np.inf
    for _ in range(int(max_iter)):
        theta = q @ p  # (M,)
        theta = np.clip(theta, eps, 1.0 - eps)

        qp = (q[:, None] * p)               # (K,M)
        qmp = (q[:, None] * (1.0 - p))      # (K,M)

        idx2 = (X == 2)
        idx0 = (X == 0)
        idx1 = (X == 1)

        counts = np.zeros(K, dtype=float)
        if np.any(idx2):
            counts += np.sum(2.0 * qp[:, idx2] / theta[idx2], axis=1)
        if np.any(idx0):
            counts += np.sum(2.0 * qmp[:, idx0] / (1.0 - theta[idx0]), axis=1)
        if np.any(idx1):
            counts += np.sum(qp[:, idx1] / theta[idx1] + qmp[:, idx1] / (1.0 - theta[idx1]), axis=1)

        q_new = counts / (2.0 * M)
        q_new = _normalize(q_new)

        ll = loglik_AM_diploid(X, q_new, p)
        if abs(ll - prev) < tol:
            q = q_new
            break
        prev = ll
        q = q_new

    return q


def logsumexp(a, axis=None, keepdims=False):
    a = np.asarray(a, dtype=float)
    amax = np.max(a, axis=axis, keepdims=True)
    out = amax + np.log(np.sum(np.exp(a - amax), axis=axis, keepdims=True))
    if not keepdims:
        out = np.squeeze(out, axis=axis)
    return out


def create_sample_pbekannt(M, K, p, q, rng=None):
    """
    Simulate X under Admixture truth with known p and q:
      Z_m iid ~ q
      X_m | Z_m=k ~ Bernoulli(p[k,m])
    Accepts p as (K,M) or (M,K).
    """
    if rng is None:
        rng = np.random.default_rng()
    q = _normalize(q)
    p = _ensure_KM(p, K, M)

    Z = rng.choice(np.arange(K), size=M, p=q)
    X = rng.binomial(1, p[Z, np.arange(M)])
    return X.astype(int)


def create_sample_pbekannt_diploid(M, K, p, q, rng=None):
    """Simulate diploid genotypes X in {0,1,2} under Admixture truth.

    Marginally, X_m ~ Binomial(2, theta_m) where theta_m = sum_k q_k p[k,m].
    """
    if rng is None:
        rng = np.random.default_rng()
    q = _normalize(q)
    p = _ensure_KM(p, K, M)

    theta = q @ p
    eps = 1e-12
    theta = np.clip(theta, eps, 1 - eps)

    X = rng.binomial(2, theta)
    return X.astype(int)


def loglik_LM(X, q, p, d, r):
    """
    Linkage model log-likelihood via Grid_Search_Linkage.forward_log_likelihood.
    Accepts p as (K,M) or (M,K).
    """
    X = _ensure_1d_X(X)
    M = len(X)
    q = _normalize(q)
    K = len(q)
    p = _ensure_KM(p, K, M)
    d = np.asarray(d, dtype=float)
    if d.shape[0] != M:
        raise ValueError(f"d must have length M={M}, got {d.shape[0]}")

    return float(grid_l.forward_log_likelihood(X, q, p, d, float(r)))


def loglik_LM_diploid(X, q, p, d, r):
    """Diploid linkage model log-likelihood via Grid_Search_Linkage.forward_log_likelihood_diploid."""
    X = _ensure_1d_X(X)
    M = len(X)
    q = _normalize(q)
    K = len(q)
    p = _ensure_KM(p, K, M)
    d = np.asarray(d, dtype=float)
    if d.shape[0] != M:
        raise ValueError(f"d must have length M={M}, got {d.shape[0]}")
    return float(grid_l.forward_log_likelihood_diploid(X, q, p, d, float(r)))


# ---------- LRT ----------
def likelihood_ratio_test(logL_restricted, logL_unrestricted, df=1, alpha=0.05):
    LR_stat = 2.0 * (float(logL_unrestricted) - float(logL_restricted))
    p_value = 1.0 - chi2.cdf(LR_stat, df)
    critical_value = chi2.ppf(1.0 - alpha, df)
    reject = bool(LR_stat > critical_value)
    return {
        "LR_statistic": float(LR_stat),
        "p_value": float(p_value),
        "critical_value": float(critical_value),
        "reject_H0": reject,
    }


# ---------- Main test summary ----------
def test_summary(X, p, d, K, q, r, alpha=0.05, r_grid=None, q_step=0.02, n_em_iter=500, ploidy: int = 1):
    """
    Compute MLEs under:
      H1 (Linkage): (q_hat_LM, r_hat) via grid search
      H0 (Admixture): q_hat_AM via EM
    then perform LRT with df=1.

    Returns dict with decision and estimates.
    """
    X = _ensure_1d_X(X)
    M = len(X)
    p = _ensure_KM(p, K, M)
    d = np.asarray(d, dtype=float)
    if d.shape[0] != M:
        raise ValueError(f"d must have length M={M}, got {d.shape[0]}")

    if r_grid is None:
        r_grid = np.linspace(0.0, 2.0, 100)

    if int(ploidy) not in (1, 2):
        raise ValueError("ploidy must be 1 (haploid) or 2 (diploid).")

    # MLE under Linkage (unrestricted)
    q_hat_LM, r_hat, logL_LM = grid_l.grid_search_q_r(
        X=X, p=p, r_range=r_grid, K=K, d_t=d, q_step=q_step, diploid=(int(ploidy) == 2)
    )

    q_hat_LM = _normalize(q_hat_LM)

    # MLE under Admixture (restricted)
    if int(ploidy) == 1:
        q_hat_AM = mle_q_admixture_EM(X, p, K, max_iter=n_em_iter)
        logL_AM = loglik_AM(X, q_hat_AM, p)
    else:
        q_hat_AM = mle_q_admixture_EM_diploid(X, p, K, max_iter=n_em_iter)
        logL_AM = loglik_AM_diploid(X, q_hat_AM, p)
    q_hat_AM = _normalize(q_hat_AM)

    out = likelihood_ratio_test(logL_AM, logL_LM, df=1, alpha=alpha)
    out.update({
        "q_hat_LM": q_hat_LM,
        "r_hat": float(r_hat),
        "q_hat_AM": q_hat_AM,
        "logL_LM": float(logL_LM),
        "logL_AM": float(logL_AM),
        "ploidy": int(ploidy),
    })
    return out

