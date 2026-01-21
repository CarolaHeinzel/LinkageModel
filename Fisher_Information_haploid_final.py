import numpy as np
import math
import itertools

def transition_prob(z_prev, z_next, d, r, q):
    """
    Transition probability for HMM:
    P(Z_m = k | Z_{m-1} = k_tilde, r, q)
    """
    if z_next == z_prev:
        return math.exp(-d * r) + (1 - math.exp(-d * r)) * q[z_next]
    else:
        return (1 - math.exp(-d * r)) * q[z_next]

def emission_prob(x_i, z, q, p, i):
    """
    Emission probability for HMM:
    P(X_m = x | Z_m = z) = Ber(q_z * p_{z, m})
    """
    prob = q[z] * p[z][i]
    return prob**x_i * (1 - prob)**(1 - x_i)

def transition_prob_der(z_prev, z_next, d, r, q, param_index, q_index=None):
    # Derivative w.r.t. r
    if param_index == 'r':
        if z_next == z_prev:
            return -d * math.exp(-d * r) + d * math.exp(-d * r) * q[z_next]
        else:
            return d * math.exp(-d * r) * q[z_next]
    # Derivative w.r.t. q_j
    elif param_index == 'q' and q_index is not None:
        if z_next == q_index:
            if z_next == z_prev:
                return 1 - math.exp(-d * r)
            else:
                return 1 - math.exp(-d * r)
        else:
            return 0.0
    else:
        return 0.0

def emission_prob_der(x_i, z, q, p, i, param_index, q_index=None):
    # Derivative w.r.t. q_j
    if param_index == 'q' and q_index is not None:
        if z == q_index:
            p_zi = p[z][i]
            prob = q[z] * p_zi
            if x_i == 1:
                return p_zi * prob**(x_i - 1) * (1 - prob)**(1 - x_i)
            else:
                return -p_zi * prob**x_i * (1 - prob)**(-x_i)
        else:
            return 0.0
    else:
        return 0.0

def forward_algorithm_derivative(x, K, q, p, d_list, r, param_index='q', q_index=0):
    M = len(x)
    alpha = np.zeros((M, K))
    alpha_der = np.zeros((M, K))
    # Initialization i = 0
    for z in range(K):
        e_prob = emission_prob(x[0], z, q, p, 0)
        if param_index == 'q' and z == q_index:
            e_der = emission_prob_der(x[0], z, q, p, 0, param_index, q_index)
        else:
            e_der = 0.0
        alpha[0, z] = q[z] * e_prob
        alpha_der[0, z] = (e_der * q[z]) + (e_prob if param_index == 'q' and z == q_index else 0.0)
    # Recursion i >= 1
    for i in range(1, M):
        for z in range(K):
            sum_alpha = 0.0
            sum_alpha_der = 0.0
            for z_prev in range(K):
                trans = transition_prob(z_prev, z, d_list[i-1], r, q)
                trans_der = transition_prob_der(z_prev, z, d_list[i-1], r, q, param_index, q_index)
                contrib = alpha[i-1, z_prev] * trans
                contrib_der = alpha_der[i-1, z_prev] * trans + alpha[i-1, z_prev] * trans_der
                sum_alpha += contrib
                sum_alpha_der += contrib_der
            e_prob = emission_prob(x[i], z, q, p, i)
            if param_index == 'q' and z == q_index:
                e_der = emission_prob_der(x[i], z, q, p, i, param_index, q_index)
            else:
                e_der = 0.0
            alpha[i, z] = sum_alpha * e_prob
            alpha_der[i, z] = sum_alpha_der * e_prob + sum_alpha * e_der
    prob = np.sum(alpha[-1])
    d_prob = np.sum(alpha_der[-1])
    return prob, d_prob

def forward_algorithm(x, K, q, p, d_list, r):
    M = len(x)
    alpha = np.zeros((M, K))
    # Initialization
    for z in range(K):
        e_prob = emission_prob(x[0], z, q, p, 0)
        alpha[0, z] = q[z] * e_prob
    # Recursion
    for i in range(1, M):
        for z in range(K):
            sum_alpha = 0.0
            for z_prev in range(K):
                trans = transition_prob(z_prev, z, d_list[i-1], r, q)
                sum_alpha += alpha[i-1, z_prev] * trans
            e_prob = emission_prob(x[i], z, q, p, i)
            alpha[i, z] = sum_alpha * e_prob
    return np.sum(alpha[-1])

def fisher_information_matrix(K, q, p, d_list, r):
    # Fisher information for all q and r
    n_params = K + 1  # q_0,...,q_{K-1}, r
    FI = np.zeros((n_params, n_params))
    M = len(p[0])
    for x in itertools.product([0, 1], repeat=M):
        p_x = forward_algorithm(x, K, q, p, d_list, r)
        grads = []
        # Derivative w.r.t. each q
        for k in range(K):
            _, dq = forward_algorithm_derivative(x, K, q, p, d_list, r, param_index='q', q_index=k)
            grads.append(dq)
        # Derivative w.r.t. r
        _, dr = forward_algorithm_derivative(x, K, q, p, d_list, r, param_index='r')
        grads.append(dr)
        grads = np.array(grads)
        FI += p_x * np.outer(grads, grads)
    return FI

# Example usage
K = 2
M = 8
q = [0.1, 0.9]
p = [
     [0.9, 0.5, 0.6, 0.9, 0.9, 0.8, 0.95, 0.6],
     [0.1, 0.5, 0.7, 0.1, 0.2, 0.3, 0.01, 0.5]
    
]
r = 0.5
d_list = [10, 10, 1, 10, 1, 10, 1, 10]
FI = fisher_information_matrix(K, q, p, d_list, r)
print("Fisher information matrix:")
print(FI)
