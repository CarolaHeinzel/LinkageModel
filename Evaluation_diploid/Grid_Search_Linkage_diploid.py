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


def forward_log_likelihood_diploid(X, q, p, d, r):
    """Diploid forward log-likelihood for genotype X in {0,1,2}.

    Diploid model: two independent hidden chains (haplotypes) with the same
    linkage transition, and genotype emissions under Hardy–Weinberg conditional
    on the two ancestries.

      Z0^(a) ~ q,  Z0^(b) ~ q
      transition per haplotype: T[j,k] = stay*1_{k=j} + (1-stay)*q[k]
      X_t | (Z_t^(a)=i, Z_t^(b)=j) ~ Binomial(2, p_i,t and p_j,t combined)

    Implemented as a forward algorithm on the K^2 product state-space.
    """
    X = np.asarray(X, dtype=int)
    q = _normalize_prob_vec(q)
    p = np.asarray(p, dtype=float)
    d = np.asarray(d, dtype=float)

    M = X.shape[0]
    K = q.shape[0]

    if p.shape == (M, K):
        p = p.T
    if p.shape != (K, M):
        raise AssertionError(f"p must have shape (K,M)=({K},{M}) or (M,K)=({M},{K}), got {p.shape}")
    if d.shape[0] != M:
        raise AssertionError(f"d must have length M={M}, got {d.shape[0]}")

    if np.any((X < 0) | (X > 2)):
        raise ValueError("Diploid genotype X must take values in {0,1,2}.")

    eps = 1e-300

    def emit_matrix(t: int):
        pt = np.clip(p[:, t], eps, 1.0 - eps)
        if X[t] == 2:
            return np.outer(pt, pt)
        if X[t] == 0:
            qt = 1.0 - pt
            return np.outer(qt, qt)
        # X[t] == 1
        qt = 1.0 - pt
        return np.outer(pt, qt) + np.outer(qt, pt)

    # alpha is KxK over pair-states; scale[t] is normalization
    alpha = np.outer(q, q) * emit_matrix(0)
    scale0 = float(alpha.sum())
    if scale0 <= 0:
        return -np.inf
    alpha /= scale0
    loglik = np.log(scale0)

    for t in range(1, M):
        stay = float(np.exp(-d[t] * r))
        switch = 1.0 - stay

        # single-haplotype transition matrix T
        T = switch * np.tile(q, (K, 1))
        T += stay * np.eye(K)

        # propagate pair distribution: alpha_pred[k,l] = sum_{i,j} alpha[i,j] T[i,k] T[j,l]
        alpha = T.T @ alpha @ T

        # multiply emission and renormalize
        alpha *= emit_matrix(t)
        sc = float(alpha.sum())
        if sc <= 0:
            return -np.inf
        alpha /= sc
        loglik += np.log(sc)

    return float(loglik)


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


def grid_search_q_r(X, p, r_range, K, d_t, q_step=0.05, use_dirichlet_if_large=True, dirichlet_samples=2000, rng=None, diploid: bool = False):
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

    ll_fn = forward_log_likelihood_diploid if diploid else forward_log_likelihood

    for q in q_candidates:
        for r in r_range:
            ll = ll_fn(X, q, p, d_t, float(r))
            if ll > best_loglik:
                best_loglik = ll
                best_q = np.asarray(q, dtype=float)
                best_r = float(r)

    return best_q, best_r, float(best_loglik)
