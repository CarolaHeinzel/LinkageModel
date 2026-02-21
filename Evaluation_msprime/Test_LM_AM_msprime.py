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



# =========================
# Diploid extensions
# =========================

from scipy.optimize import minimize


def loglik_AM_diploid(X, q, p):
    """Admixture log-likelihood for diploid genotypes X in {0,1,2}.

    theta_m = sum_k q_k p[k,m]
    X_m ~ Binomial(n=2, p=theta_m)
    """
    X = _ensure_1d_X(X)
    M = len(X)
    q = _normalize(q)
    K = len(q)
    p = _ensure_KM(p, K, M)

    theta = q @ p
    eps = 1e-12
    theta = np.clip(theta, eps, 1.0 - eps)

    # log Binomial pmf for n=2
    # x=0: 2*log(1-theta)
    # x=1: log(2) + log(theta) + log(1-theta)
    # x=2: 2*log(theta)
    log2 = np.log(2.0)
    ll = 0.0
    for xm, th in zip(X, theta):
        if xm == 0:
            ll += 2.0 * np.log(1.0 - th)
        elif xm == 2:
            ll += 2.0 * np.log(th)
        else:
            ll += log2 + np.log(th) + np.log(1.0 - th)
    return float(ll)


def create_sample_pbekannt_diploid(M, K, p, q, rng=None):
    """Simulate diploid genotypes under Admixture truth with known p and q.

    theta_m = sum_k q_k p[k,m]
    X_m ~ Binomial(2, theta_m)
    """
    if rng is None:
        rng = np.random.default_rng()
    q = _normalize(q)
    p = _ensure_KM(p, K, M)
    theta = q @ p
    eps = 1e-12
    theta = np.clip(theta, eps, 1.0 - eps)
    X = rng.binomial(2, theta)
    return X.astype(int)


def mle_q_admixture_EM_diploid(X, p, K, q_init=None, max_iter=500, tol=1e-10):
    """EM for admixture proportions q with known p, diploid genotypes X in {0,1,2}.

    Latent model: each of the 2 alleles has ancestry ~ q and allele ~ Bernoulli(p).
    Observed genotype is the sum of the 2 alleles.

    Update uses expected ancestry counts per allele:
      q_new[k] = (1/M) * sum_m w_{m,k}
    where w_{m,k} = P(Z_{m,allele}=k | X_m).
    """
    X = _ensure_1d_X(X)
    M = len(X)
    p = _ensure_KM(p, K, M)

    eps = 1e-12
    p = np.clip(p, eps, 1.0 - eps)

    if q_init is None:
        q = np.ones(K) / K
    else:
        q = _normalize(q_init)

    prev = -np.inf
    for _ in range(max_iter):
        theta = q @ p  # (M,)
        theta = np.clip(theta, eps, 1.0 - eps)

        w = np.zeros((K, M), dtype=float)

        # x=0
        idx0 = (X == 0)
        if np.any(idx0):
            denom = 1.0 - theta[idx0]
            denom = np.clip(denom, eps, None)
            w[:, idx0] = (q[:, None] * (1.0 - p[:, idx0])) / denom[None, :]

        # x=2
        idx2 = (X == 2)
        if np.any(idx2):
            denom = theta[idx2]
            denom = np.clip(denom, eps, None)
            w[:, idx2] = (q[:, None] * p[:, idx2]) / denom[None, :]

        # x=1
        idx1 = (X == 1)
        if np.any(idx1):
            th = theta[idx1]
            denom = 2.0 * th * (1.0 - th)
            denom = np.clip(denom, eps, None)
            numer = q[:, None] * (p[:, idx1] * (1.0 - th)[None, :] + (1.0 - p[:, idx1]) * th[None, :])
            w[:, idx1] = numer / denom[None, :]

        # Normalize (numerical safety)
        w_sum = w.sum(axis=0, keepdims=True)
        w_sum = np.clip(w_sum, eps, None)
        w = w / w_sum

        q_new = w.mean(axis=1)
        q_new = _normalize(q_new)

        ll = loglik_AM_diploid(X, q_new, p)
        if abs(ll - prev) < tol:
            q = q_new
            break
        prev = ll
        q = q_new

    return q


def loglik_LM_diploid(X, q, p, d, r):
    """Diploid linkage-model log-likelihood via Grid_Search_Linkage.forward_log_likelihood_diploid."""
    X = _ensure_1d_X(X)
    M = len(X)
    q = _normalize(q)
    K = len(q)
    p = _ensure_KM(p, K, M)
    d = np.asarray(d, dtype=float)
    if d.shape[0] != M:
        raise ValueError(f"d must have length M={M}, got {d.shape[0]}")
    return float(grid_l.forward_log_likelihood_diploid(X, q, p, d, float(r)))


def _softmax(u):
    u = np.asarray(u, dtype=float)
    u = u - np.max(u)
    e = np.exp(u)
    return e / np.sum(e)


def mle_q_r_linkage_opt(X, p, d, K, ploidy=1, q_init=None, r_init=1.0,
                        r_bounds=(1e-6, 50.0), maxiter=200):
    """Numerical MLE for (q,r) under the linkage model.

    Uses a softmax parameterization for q and log-parameterization for r.

    ploidy=1: X in {0,1}, uses forward_log_likelihood
    ploidy=2: X in {0,1,2}, uses forward_log_likelihood_diploid
    """
    X = _ensure_1d_X(X)
    M = len(X)
    p = _ensure_KM(p, K, M)
    d = np.asarray(d, dtype=float)

    if q_init is None:
        q0 = np.ones(K) / K
    else:
        q0 = _normalize(q_init)

    # init params
    u0 = np.log(q0 + 1e-12)
    lr0 = np.log(float(r_init) + 1e-12)
    x0 = np.concatenate([u0, [lr0]])

    # bounds
    lo_r = float(r_bounds[0])
    hi_r = float(r_bounds[1])
    bnds = [(-20.0, 20.0)] * K + [(np.log(lo_r + 1e-12), np.log(hi_r + 1e-12))]

    if ploidy == 2:
        ll_fun = lambda q, r: grid_l.forward_log_likelihood_diploid(X, q, p, d, r)
    else:
        ll_fun = lambda q, r: grid_l.forward_log_likelihood(X, q, p, d, r)

    def obj(v):
        u = v[:K]
        lr = v[K]
        q = _softmax(u)
        r = float(np.exp(lr))
        ll = ll_fun(q, r)
        return -ll if np.isfinite(ll) else 1e100

    res = minimize(obj, x0, method='L-BFGS-B', bounds=bnds, options={'maxiter': int(maxiter)})

    u_hat = res.x[:K]
    lr_hat = res.x[K]
    q_hat = _softmax(u_hat)
    r_hat = float(np.exp(lr_hat))

    logL = -float(res.fun)
    return q_hat, r_hat, logL


# Override/extend test_summary with ploidy support (keeps old API compatible)
def test_summary(X, p, d, K, q=None, r=None, alpha=0.05, r_grid=None, q_step=0.02, n_em_iter=500, ploidy=1):
    """Compute the LRT comparing Admixture (H0) vs Linkage (H1).

    ploidy=1:
      X in {0,1}
      H0: iid Bernoulli with theta_m = sum_k q_k p[k,m]
      H1: linkage HMM with parameters (q,r)

    ploidy=2:
      X in {0,1,2}
      H0: iid Binomial(2, theta_m)
      H1: diploid linkage model (two haplotypes) with parameters (q,r)

    Returns dict with LR statistic, p-value, decision, and MLEs.
    """
    X = _ensure_1d_X(X)
    M = len(X)
    p = _ensure_KM(p, K, M)
    d = np.asarray(d, dtype=float)
    if d.shape[0] != M:
        raise ValueError(f"d must have length M={M}, got {d.shape[0]}")

    # --- H1 (Linkage): numerical MLE over (q,r) ---
    # Use AM MLE as a good start for q
    if ploidy == 2:
        q_start = mle_q_admixture_EM_diploid(X, p, K, max_iter=n_em_iter)
    else:
        q_start = mle_q_admixture_EM(X, p, K, max_iter=n_em_iter)

    q_hat_LM, r_hat, logL_LM = mle_q_r_linkage_opt(
        X=X, p=p, d=d, K=K, ploidy=ploidy, q_init=q_start, r_init=1.0, r_bounds=(1e-6, 50.0)
    )

    # --- H0 (Admixture): MLE over q ---
    if ploidy == 2:
        q_hat_AM = mle_q_admixture_EM_diploid(X, p, K, max_iter=n_em_iter)
        logL_AM = loglik_AM_diploid(X, q_hat_AM, p)
    else:
        q_hat_AM = mle_q_admixture_EM(X, p, K, max_iter=n_em_iter)
        logL_AM = loglik_AM(X, q_hat_AM, p)

    out = likelihood_ratio_test(logL_AM, logL_LM, df=1, alpha=alpha)
    out.update({
        'q_hat_LM': q_hat_LM,
        'r_hat': float(r_hat),
        'q_hat_AM': q_hat_AM,
        'logL_LM': float(logL_LM),
        'logL_AM': float(logL_AM),
        'ploidy': int(ploidy),
    })
    return out


# ---- Faster override: reuse q_hat_AM as warm start for linkage MLE ----
def test_summary(X, p, d, K, q=None, r=None, alpha=0.05, r_grid=None, q_step=0.02,
                 n_em_iter=200, ploidy=1, lm_maxiter=80, r_bounds=(1e-6, 50.0)):
    """Same API as before, but faster for large M by:
    - computing q_hat_AM once and reusing it as warm start for the linkage MLE
    - limiting linkage optimizer iterations (lm_maxiter)

    Parameters
    ----------
    n_em_iter : int
        Max EM iterations for q under Admixture.
    lm_maxiter : int
        Max optimizer iterations for (q,r) under Linkage.
    r_bounds : tuple
        Bounds for r in the linkage optimizer.
    """
    X = _ensure_1d_X(X)
    M = len(X)
    p = _ensure_KM(p, K, M)
    d = np.asarray(d, dtype=float)
    if d.shape[0] != M:
        raise ValueError(f"d must have length M={M}, got {d.shape[0]}")

    # --- H0 (Admixture): MLE over q ---
    if ploidy == 2:
        q_hat_AM = mle_q_admixture_EM_diploid(X, p, K, max_iter=n_em_iter)
        logL_AM = loglik_AM_diploid(X, q_hat_AM, p)
    else:
        q_hat_AM = mle_q_admixture_EM(X, p, K, max_iter=n_em_iter)
        logL_AM = loglik_AM(X, q_hat_AM, p)

    # --- H1 (Linkage): numerical MLE over (q,r), warm start from q_hat_AM ---
    q_hat_LM, r_hat, logL_LM = mle_q_r_linkage_opt(
        X=X, p=p, d=d, K=K, ploidy=ploidy, q_init=q_hat_AM, r_init=1.0, r_bounds=r_bounds, maxiter=lm_maxiter
    )

    out = likelihood_ratio_test(logL_AM, logL_LM, df=1, alpha=alpha)
    out.update({
        'q_hat_LM': q_hat_LM,
        'r_hat': float(r_hat),
        'q_hat_AM': q_hat_AM,
        'logL_LM': float(logL_LM),
        'logL_AM': float(logL_AM),
        'ploidy': int(ploidy),
    })
    return out


# ---- Faster override of mle_q_r_linkage_opt: precompute emissions for repeated likelihood calls ----
def mle_q_r_linkage_opt(X, p, d, K, ploidy=1, q_init=None, r_init=1.0,
                        r_bounds=(1e-6, 50.0), maxiter=80):
    """Numerical MLE for (q,r) under the linkage model with emission precomputation."""
    X = _ensure_1d_X(X)
    M = len(X)
    p = _ensure_KM(p, K, M)
    d = np.asarray(d, dtype=float)

    if q_init is None:
        q0 = np.ones(K) / K
    else:
        q0 = _normalize(q_init)

    # init params
    u0 = np.log(q0 + 1e-12)
    lr0 = np.log(float(r_init) + 1e-12)
    x0 = np.concatenate([u0, [lr0]])

    lo_r = float(r_bounds[0])
    hi_r = float(r_bounds[1])
    bnds = [(-20.0, 20.0)] * K + [(np.log(lo_r + 1e-12), np.log(hi_r + 1e-12))]

    eps = 1e-12

    if ploidy == 2:
        # Precompute emission matrices E[t,:,:]
        E = np.empty((M, K, K), dtype=float)
        for t in range(M):
            pc = p[:, t]
            p_i = pc[:, None]
            p_j = pc[None, :]
            if X[t] == 0:
                E[t] = (1.0 - p_i) * (1.0 - p_j)
            elif X[t] == 2:
                E[t] = p_i * p_j
            else:
                E[t] = p_i * (1.0 - p_j) + (1.0 - p_i) * p_j
        E = np.clip(E, eps, None)

        def ll_fast(q, r):
            # forward with alpha scaled to sum to 1
            alpha = np.outer(q, q) * E[0]
            s = alpha.sum()
            if s <= 0:
                return -np.inf
            ll = math.log(s)
            alpha /= s

            for t in range(1, M):
                stay = math.exp(-d[t] * r)
                switch = 1.0 - stay
                rowsum = alpha.sum(axis=1)
                colsum = alpha.sum(axis=0)

                predicted = (stay * stay) * alpha
                predicted += stay * switch * (np.outer(q, colsum) + np.outer(rowsum, q))
                predicted += (switch * switch) * np.outer(q, q)

                alpha = predicted * E[t]
                s = alpha.sum()
                if s <= 0:
                    return -np.inf
                ll += math.log(s)
                alpha /= s
            return float(ll)

    else:
        # Haploid: precompute emission vectors e[t,k]
        e = np.empty((M, K), dtype=float)
        for t in range(M):
            if X[t] == 1:
                e[t] = p[:, t]
            else:
                e[t] = 1.0 - p[:, t]
        e = np.clip(e, eps, None)

        def ll_fast(q, r):
            alpha = q * e[0]
            s = alpha.sum()
            if s <= 0:
                return -np.inf
            ll = math.log(s)
            alpha /= s
            for t in range(1, M):
                stay = math.exp(-d[t] * r)
                switch = 1.0 - stay
                predicted = stay * alpha + switch * q
                alpha = predicted * e[t]
                s = alpha.sum()
                if s <= 0:
                    return -np.inf
                ll += math.log(s)
                alpha /= s
            return float(ll)

    def obj(v):
        u = v[:K]
        lr = v[K]
        q = _softmax(u)
        r = float(np.exp(lr))
        ll = ll_fast(q, r)
        return -ll if np.isfinite(ll) else 1e100

    res = minimize(obj, x0, method='L-BFGS-B', bounds=bnds, options={'maxiter': int(maxiter)})

    u_hat = res.x[:K]
    lr_hat = res.x[K]
    q_hat = _softmax(u_hat)
    r_hat = float(np.exp(lr_hat))
    logL = -float(res.fun)

    return q_hat, r_hat, logL

import math

