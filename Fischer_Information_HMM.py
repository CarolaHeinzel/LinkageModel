import numpy as np
import itertools

# Hidden Markov Model Fisher Information for emission P(X_t=x|Z_t=z) = q_z p_{t,z}
# and transition as described in the prompt
# Only q and r are estimated, with constraints sum(q)=1, q>=0

def transition_prob(z_prev, z_next, d, r, q):
    if z_next == z_prev:
        return np.exp(-d * r) + (1 - np.exp(-d * r)) * q[z_next]
    else:
        return (1 - np.exp(-d * r)) * q[z_next]

def emission_prob(x_t, z, q, p, t):
    prob = q[z] * p[t, z]
    return prob if x_t == 1 else 1 - prob

def forward_algorithm(x, K, q, p, d_list, r):
    M = len(x)
    alpha = np.zeros((M, K))
    for z in range(K):
        e_prob = emission_prob(x[0], z, q, p, 0)
        alpha[0, z] = q[z] * e_prob
    for t in range(1, M):
        for z in range(K):
            sum_alpha = 0.0
            for z_prev in range(K):
                trans = transition_prob(z_prev, z, d_list[t], r, q)
                sum_alpha += alpha[t-1, z_prev] * trans
            e_prob = emission_prob(x[t], z, q, p, t)
            alpha[t, z] = sum_alpha * e_prob
    return np.sum(alpha[-1])

def forward_derivative(x, K, q, p, d_list, r, param_index, q_index=None):
    M = len(x)
    alpha = np.zeros((M, K))
    alpha_der = np.zeros((M, K))
    for z in range(K):
        e_prob = emission_prob(x[0], z, q, p, 0)
        if param_index == 'q' and z == q_index:
            e_der = p[0, z] if x[0] == 1 else -p[0, z]
        else:
            e_der = 0.0
        alpha[0, z] = q[z] * e_prob
        alpha_der[0, z] = (e_der * q[z]) + (e_prob if param_index == 'q' and z == q_index else 0.0)
    for t in range(1, M):
        for z in range(K):
            sum_alpha = 0.0
            sum_alpha_der = 0.0
            for z_prev in range(K):
                trans = transition_prob(z_prev, z, d_list[t], r, q)
                if param_index == 'r':
                    if z == z_prev:
                        trans_der = -d_list[t] * np.exp(-d_list[t] * r) + d_list[t] * np.exp(-d_list[t] * r) * q[z]
                    else:
                        trans_der = d_list[t] * np.exp(-d_list[t] * r) * q[z]
                elif param_index == 'q' and q_index is not None:
                    if z == q_index:
                        trans_der = (1 - np.exp(-d_list[t] * r))
                    else:
                        trans_der = 0.0
                else:
                    trans_der = 0.0
                contrib = alpha[t-1, z_prev] * trans
                contrib_der = alpha_der[t-1, z_prev] * trans + alpha[t-1, z_prev] * trans_der
                sum_alpha += contrib
                sum_alpha_der += contrib_der
            e_prob = emission_prob(x[t], z, q, p, t)
            if param_index == 'q' and z == q_index:
                e_der = p[t, z] if x[t] == 1 else -p[t, z]
            else:
                e_der = 0.0
            alpha[t, z] = sum_alpha * e_prob
            alpha_der[t, z] = sum_alpha_der * e_prob + sum_alpha * e_der
    prob = np.sum(alpha[-1])
    d_prob = np.sum(alpha_der[-1])
    return prob, d_prob

def fisher_information_matrix_reduced(K, q, p, d_list, r):
    # Only one free q parameter (q1), q2 = 1 - q1
    n_params = 2  # q1, r
    FI = np.zeros((n_params, n_params))
    M = p.shape[0]
    for x in itertools.product([0, 1], repeat=M):
        # q = [q1, 1-q1]
        q1 = q[0]
        q_vec = np.array([q1, 1-q1])
        # Derivatives w.r.t. q1 and r
        p_x = forward_algorithm(x, K, q_vec, p, d_list, r)
        # d/dq1: chain rule for q1 and q2
        _, dq1 = forward_derivative(x, K, q_vec, p, d_list, r, param_index='q', q_index=0)
        _, dq2 = forward_derivative(x, K, q_vec, p, d_list, r, param_index='q', q_index=1)
        dq = dq1 - dq2  # since q2 = 1 - q1
        _, dr = forward_derivative(x, K, q_vec, p, d_list, r, param_index='r')
        grads = np.array([dq, dr])
        FI += p_x * np.outer(grads, grads)
    return FI

def fisher_information_matrix_reduced_general(K, q, p, d_list, r):
    # K-1 free q parameters, q_K = 1 - sum(q_1,...,q_{K-1})
    n_params = (K-1) + 1  # (K-1) q's, 1 r
    FI = np.zeros((n_params, n_params))
    M = p.shape[0]
    for x in itertools.product([0, 1], repeat=M):
        # q_full = [q1, ..., q_{K-1}, 1-sum(q1..q_{K-1})]
        q_vec = np.zeros(K)
        q_vec[:-1] = q[:K-1]
        q_vec[-1] = 1 - np.sum(q[:K-1])
        p_x = forward_algorithm(x, K, q_vec, p, d_list, r)
        grads = []
        # Derivatives w.r.t. q1,...,q_{K-1}
        for k in range(K-1):
            _, dqk = forward_derivative(x, K, q_vec, p, d_list, r, param_index='q', q_index=k)
            _, dqK = forward_derivative(x, K, q_vec, p, d_list, r, param_index='q', q_index=K-1)
            dq = dqk - dqK  # chain rule: q_K = 1 - sum(q1..q_{K-1})
            grads.append(dq)
        # Derivative w.r.t. r
        _, dr = forward_derivative(x, K, q_vec, p, d_list, r, param_index='r')
        grads.append(dr)
        grads = np.array(grads)
        FI += p_x * np.outer(grads, grads)
    return FI

# Example usage
K = 2
M = 4
q = np.array([0.3, 0.7])
p = np.array([[0.2, 0.9], [0.3, 0.8], [0.7, 0.6], [0.1, 0.9]])  # shape (M, K)
r = 0.5
d_list = [1.0] * M
FI_reduced = fisher_information_matrix_reduced(K, q, p, d_list, r)
print("Fisher information matrix (reduced, q1 only):")
print(FI_reduced)

# Example usage for general K
#%%
K = 3
M = 4
q = np.array([0.2, 0.5, 0.3])
p = np.array([[0.6, 0.4, 0.5], [0.9, 0.3, 0.4], [0.7, 0.6, 0.7], [0.1, 0.2, 0.1]])  # shape (M, K)
r = 0.5
d_list = [1.0] * M
FI_general = fisher_information_matrix_reduced_general(K, q, p, d_list, r)
print("Fisher information matrix (reduced, general K):")
print(FI_general)

