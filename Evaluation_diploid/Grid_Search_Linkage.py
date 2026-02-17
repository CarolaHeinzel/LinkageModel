# Stable forward log-likelihood + grid search over q (simplex grid) and r.
# Finds the MLE by using grid search
import numpy as np


def _normalize_prob_vec(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    s = v.sum()
    if s <= 0:
        raise ValueError("Probability vector must have positive sum.")
    return v / s


def forward_log_likelihood(X, q, p, d, r):
    """
    Compute log P(X_0,...,X_{M-1} | r, q, p, d) using a scaled Forward Algorithm.

    Model:
      Z_0 ~ q
      P(Z_t=k | Z_{t-1}=j) = stay*1_{k=j} + (1-stay)*q[k], stay=exp(-d[t]*r)
      X_t | Z_t=k ~ Bernoulli(p[k,t])

    Parameters
    ----------
    X : array, shape (M,)
    q : array, shape (K,)
    p : array, shape (K,M) OR (M,K)
    d : array, shape (M,)   (d[0] may be 0; first transition uses d[1])
    r : float

    Returns
    -------
    float : log-likelihood
    """
    X = np.asarray(X, dtype=int)
    q = _normalize_prob_vec(q)
    p = np.asarray(p, dtype=float)
    d = np.asarray(d, dtype=float)

    M = X.shape[0]
    K = q.shape[0]

    # Accept p as (K,M) or (M,K)
    if p.shape == (M, K):
        p = p.T
    if p.shape != (K, M):
        raise AssertionError(f"p must have shape (K,M)=({K},{M}) or (M,K)=({M},{K}), got {p.shape}")
    if d.shape[0] != M:
        raise AssertionError(f"d must have length M={M}, got {d.shape[0]}")

    # alpha[t,k] scaled; scale[t] stores scaling factor
    alpha = np.zeros(K, dtype=float)
    scale = np.zeros(M, dtype=float)

    # emission at t
    def emit(k, t):
        pk = p[k, t]
        return pk if X[t] == 1 else (1.0 - pk)

    # init
    for k in range(K):
        alpha[k] = q[k] * emit(k, 0)

    scale[0] = alpha.sum()
    if scale[0] <= 0:
        return -np.inf
    alpha /= scale[0]

    # recursion
    for t in range(1, M):
        stay = np.exp(-d[t] * r)
        switch = 1.0 - stay

        # because sum(alpha)=1 after scaling:
        # predicted[k] = stay*alpha_prev[k] + switch*q[k]
        predicted = stay * alpha + switch * q

        # multiply emissions
        for k in range(K):
            alpha[k] = predicted[k] * emit(k, t)

        scale[t] = alpha.sum()
        if scale[t] <= 0:
            return -np.inf
        alpha /= scale[t]

    return float(np.sum(np.log(scale)))


def simplex_grid(K, step=0.05):
    """
    Generate a grid of probability vectors on the K-simplex.
    Entries are multiples of `step` and sum to 1.

    WARNING: size grows combinatorially with K and 1/step.
    """
    n = int(round(1.0 / step))
    if not np.isclose(n * step, 1.0):
        raise ValueError("step must divide 1 exactly (e.g., 0.1, 0.05, 0.02).")

    qs = []

    def rec(remaining, k_left, prefix):
        if k_left == 1:
            qs.append(np.array(prefix + [remaining], dtype=float) * step)
            return
        for i in range(remaining + 1):
            rec(remaining - i, k_left - 1, prefix + [i])

    rec(n, K, [])
    return qs


def dirichlet_candidates(K, n_samples=2000, alpha=1.0, rng=None):
    """
    Random candidates for q via Dirichlet distribution (useful for large K).
    """
    if rng is None:
        rng = np.random.default_rng()
    alpha_vec = np.ones(K) * float(alpha)
    return [rng.dirichlet(alpha_vec) for _ in range(int(n_samples))]


def grid_search_q_r(X, p, r_range, K, d_t, q_step=0.05, use_dirichlet_if_large=True, dirichlet_samples=2000, rng=None):
    """
    Grid search over q and r.

    Parameters
    ----------
    X : array (M,)
    p : array (K,M) or (M,K)
    r_range : iterable of r values
    K : int
    d_t : array/list length M
    q_step : float
        Resolution for simplex grid (only used when K is small).
    use_dirichlet_if_large : bool
        If True and K is large, use random Dirichlet samples for q.
    dirichlet_samples : int
        Number of random q samples when using Dirichlet.
    rng : np.random.Generator or None

    Returns
    -------
    best_q, best_r, best_loglik
    """
    if rng is None:
        rng = np.random.default_rng()

    X = np.asarray(X, dtype=int)
    d_t = np.asarray(d_t, dtype=float)

    # Choose q candidates
    if use_dirichlet_if_large and (K >= 6 or int(round(1.0 / q_step)) > 50):
        q_candidates = dirichlet_candidates(K, n_samples=dirichlet_samples, alpha=1.0, rng=rng)
    else:
        q_candidates = simplex_grid(K, step=q_step)

    best_loglik = -np.inf
    best_q = None
    best_r = None

    for q in q_candidates:
        for r in r_range:
            ll = forward_log_likelihood(X, q, p, d_t, float(r))
            if ll > best_loglik:
                best_loglik = ll
                best_q = np.asarray(q, dtype=float)
                best_r = float(r)

    return best_q, best_r, float(best_loglik)
