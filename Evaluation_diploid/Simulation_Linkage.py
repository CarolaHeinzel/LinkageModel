# Simulation of a linkage-style HMM:
#   Z_0 ~ q
#   P(Z_t = k | Z_{t-1} = j) = stay * 1_{k=j} + (1-stay) * q[k]
#   stay = exp(-d_t * r)
#   X_t | (Z_t=k) ~ Bernoulli(p[k, t])

import numpy as np


def _normalize_prob_vec(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    s = v.sum()
    return v / s


def create_random_matrix(M: int, K: int, low: float = 0.05, high: float = 0.95, rng=None) -> np.ndarray:
    """
    Create emission probabilities p with shape (K, M).
    p[k, t] = P(X_t = 1 | Z_t = k)

    Parameters
    ----------
    M : int
        Number of markers/time points.
    K : int
        Number of hidden states.
    low, high : float
        Range for random probabilities.
    rng : np.random.Generator or None
        Random generator for reproducibility.

    Returns
    -------
    p : np.ndarray, shape (K, M)
    """
    if rng is None:
        rng = np.random.default_rng()
    p = rng.uniform(low, high, size=(K, M))
    return p


def create_transition_matrix(K: int, q: np.ndarray, r: float, d_t: float) -> np.ndarray:
    """
    Transition matrix for the linkage model.

    Q[j, k] = P(Z_t = k | Z_{t-1} = j)

    stay = exp(-d_t * r)
    switch = 1 - stay
    Q[j, k] = stay * 1_{k=j} + switch * q[k]
    """
    q = _normalize_prob_vec(q)
    stay = float(np.exp(-d_t * r))
    switch = 1.0 - stay

    Q = np.zeros((K, K), dtype=float)
    for j in range(K):
        for k in range(K):
            Q[j, k] = (stay if k == j else 0.0) + switch * q[k]
    # Numerical safety
    Q = np.clip(Q, 0.0, 1.0)
    Q /= Q.sum(axis=1, keepdims=True)
    return Q


def simulate_markov_chain(K: int,
                          q: np.ndarray,
                          r: float,
                          steps: int,
                          d_values,
                          p: np.ndarray,
                          rng=None):
    """
    Simulate observations X_t (binary) from the linkage HMM.

    Parameters
    ----------
    K : int
        Number of hidden states.
    q : array, shape (K,)
        Initial distribution (and also used as switch-to distribution).
    r : float
        Recombination rate.
    steps : int
        Number of markers/time points (M).
    d_values : list/array, length steps
        Distances (d_values[0] can be 0; it's ignored in the first transition).
    p : array, shape (K, steps) or (steps, K)
        Emission probabilities.
    rng : np.random.Generator or None

    Returns
    -------
    X : np.ndarray, shape (steps,)
        Simulated binary observations.
    Q_last : np.ndarray, shape (K, K)
        Transition matrix at the last step (or None if steps < 2).
    """
    if rng is None:
        rng = np.random.default_rng()

    q = _normalize_prob_vec(q)
    d_values = np.asarray(d_values, dtype=float)
    if len(d_values) != steps:
        raise ValueError(f"d_values must have length steps={steps}, got {len(d_values)}")

    p = np.asarray(p, dtype=float)
    # Accept p as (K, steps) OR (steps, K)
    if p.shape == (steps, K):
        p = p.T
    if p.shape != (K, steps):
        raise ValueError(f"p must have shape (K,steps)=({K},{steps}) or (steps,K)=({steps},{K}), got {p.shape}")

    # Initial hidden state
    z = rng.choice(np.arange(K), p=q)
    x0 = rng.binomial(1, p[z, 0])

    X = np.zeros(steps, dtype=int)
    X[0] = int(x0)

    Q_last = None
    for t in range(1, steps):
        Q = create_transition_matrix(K, q, r, d_values[t])
        z = rng.choice(np.arange(K), p=Q[z])
        X[t] = int(rng.binomial(1, p[z, t]))
        Q_last = Q

    return X, Q_last
