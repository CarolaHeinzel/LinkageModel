# Implements a statistical test to compare the Linkage Model to the Admixture Model

import numpy as np
from scipy.stats import chi2
import sys
import Grid_Search_Linkage as grid_l


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
def test_summary(X, p, d, K, q, r, alpha=0.05, r_grid=None, q_step=0.02, n_em_iter=500):
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

    # MLE under Linkage (unrestricted)
    q_hat_LM, r_hat, logL_LM = grid_l.grid_search_q_r(
        X=X, p=p, r_range=r_grid, K=K, d_t=d, q_step=q_step
    )
    q_hat_LM = _normalize(q_hat_LM)

    # MLE under Admixture (restricted)
    q_hat_AM = mle_q_admixture_EM(X, p, K, max_iter=n_em_iter)
    q_hat_AM = _normalize(q_hat_AM)

    logL_AM = loglik_AM(X, q_hat_AM, p)

    out = likelihood_ratio_test(logL_AM, logL_LM, df=1, alpha=alpha)
    out.update({
        "q_hat_LM": q_hat_LM,
        "r_hat": float(r_hat),
        "q_hat_AM": q_hat_AM,
        "logL_LM": float(logL_LM),
        "logL_AM": float(logL_AM),
    })
    return out

