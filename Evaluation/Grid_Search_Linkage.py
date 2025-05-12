
import numpy as np

# Maximizes the Function

def compute_likelihood_single_chromosome(X, q, p, d, r):
    """
    Computes the likelihood P(X_1,...,X_M | r, q, p, d) using the Forward Algorithm.
    
    Parameters:
    X : np.ndarray of shape (M,)       - Observations (binary vector)
    q : np.ndarray of shape (K,)       - Initial state distribution
    p : np.ndarray of shape (K, M)     - Bernoulli parameter matrix
    d : np.ndarray of shape (M,)       - Distances between markers (d[0] ignored)
    r : float                          - Recombination rate

    Returns:
    float - Total likelihood
    """

    M = len(X)        # number of markers
    K = len(q)        # number of hidden states

    # Initialize forward probabilities alpha
    alpha = np.zeros((M, K))
    
    # Initial step: alpha_1(k) = P(Z_1 = k) * P(X_1 | Z_1 = k)
    for k in range(K):
        emission_prob = q[k] * p[k, 0] if X[0] == 1 else 1 - q[k] * p[k, 0]
        alpha[0, k] = q[k] * emission_prob

    # Recursive step
    for m in range(1, M):
        for k in range(K):
            sum_prob = 0.0
            for k_prev in range(K):
                if k == k_prev:
                    trans_prob = np.exp(-d[m] * r) + (1 - np.exp(-d[m] * r)) * q[k_prev]
                else:
                    trans_prob = (1 - np.exp(-d[m] * r)) * q[k_prev]
                sum_prob += alpha[m - 1, k_prev] * trans_prob

            emission_prob = q[k] * p[k, m] if X[m] == 1 else 1 - q[k] * p[k, m]
            alpha[m, k] = sum_prob * emission_prob

    # Total likelihood = sum over final alphas
    return np.sum(alpha[-1])

def grid_search(X, p,  r_range, K, d_t):
    """
    Führt eine Grid-Search über die Parameter q und r durch, um die Kombination mit der höchsten Log-Likelihood zu finden.
    
    Parameters:
    - X: Beobachtungsdaten (M x T Matrix)
    - r_range: Bereich für r, z.B. [0, 1, 2, ..., 100]
    - q_range: Bereich für q_1, z.B. [0.01, 0.02, ..., 1.0]
    - K: Anzahl der Zustände
    - d_t: Zeitabhängiger Parameter
    
    Returns:
    - best_q_1: Der Wert von q_1 mit der höchsten Log-Likelihood
    - best_r: Der Wert von r mit der höchsten Log-Likelihood
    - best_log_likelihood: Die höchste Log-Likelihood
    """
    best_log_likelihood = -np.inf  # Startwert für den besten Wert
    best_q_1 = None
    best_r = None
    q_range = np.linspace(0,100,101)
    # Iteriere über alle möglichen Kombinationen von q_1 und r
    for x in q_range:
        q_1 = x * 0.01  # Berechne q_1
        #print(q_1)
        for r in r_range:

            # Berechne die Log-Likelihood für dieses q_1 und r
            current_log_likelihood = compute_likelihood_single_chromosome(X, [q_1,1-q_1], p, d_t, r)
            
            # Speichere die Kombination mit der höchsten Log-Likelihood
            if current_log_likelihood > best_log_likelihood:
                best_log_likelihood = current_log_likelihood
                best_q_1 = q_1
                best_r = r
          
    return best_q_1, best_r, best_log_likelihood


#X, p, d, r_range, q_range, K, d_t
#res = grid_search(np.array(states_visited), p.T, np.linspace(0,2,30), K, d_values)
#print(res)