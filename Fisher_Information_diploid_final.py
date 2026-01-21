import numpy as np
import math
import itertools

# Transition Probability
# haploid
def transition_prob(z_prev, z_next, d, r, q):
    if z_next == z_prev:
        return math.exp(-d * r) + (1 - math.exp(-d * r)) * q[z_next]
    else:
        return (1 - math.exp(-d * r)) * q[z_next]

# Emission probability
# haploid
def emission_prob(x_i, z, q, p, i):
    prob = q[z] * p[z][i]
    return prob**x_i * (1 - prob)**(1 - x_i)

# Derivative of the Transmission probability 
# haploid
def transition_prob_der(z_prev, z_next, d, r, q, param_index, q_index=None):
    if param_index == 'r':
        if z_next == z_prev:
            return -d * math.exp(-d * r) + d * math.exp(-d * r) * q[z_next]
        else:
            return d * math.exp(-d * r) * q[z_next]
    elif param_index == 'q' and q_index is not None:
        if z_next == q_index:
            return 1 - math.exp(-d * r)
        else:
            return 0.0
    else:
        return 0.0
# Derivative of the Emission probability 
# haploid
def emission_prob_der(x_i, z, q, p, i, param_index, q_index=None):
    if param_index == 'q' and q_index is not None:
        if z == q_index:
            p_zi = p[z][i]
            if x_i == 1:
                return p_zi 
            else:
                return -p_zi 
        else:
            return 0.0
    else:
        return 0.0

# Calculates the likelihood of the data
# given all other parameters
def forward_algorithm(x, K, q, p, d_list, r):
    M = len(x)
    alpha = np.zeros((M, K))
    # initializes alpha
    for z in range(K):
        e_prob = emission_prob(x[0], z, q, p, 0)
        alpha[0, z] = q[z] * e_prob
    # calculates alpha_1,..., alpha_M
    for i in range(1, M):
        for z in range(K):
            sum_alpha = 0.0
            for z_prev in range(K):
                trans = transition_prob(z_prev, z, d_list[i-1], r, q)
                sum_alpha += alpha[i-1, z_prev] * trans
            e_prob = emission_prob(x[i], z, q, p, i)
            alpha[i, z] = sum_alpha * e_prob
    return np.sum(alpha[-1])

# Recursive Calculation of the derivation
def forward_algorithm_derivative(x, K, q, p, d_list, r, param_index='q', q_index=0):
    """
    Computes the forward probability and its derivative with respect to a parameter.

    Args:
        x: Observed data sequence (list of 0s and 1s).
        K: Number of hidden states.
        q: Initial state distribution (list of length K).
        p: Emission probabilities (list of lists; shape K x M).
        d_list: Genetic distances between positions.
        r: Recombination rate.
        param_index: 'q' or 'r' – the parameter to differentiate with respect to.
        q_index: Index of q if param_index == 'q'.

    Returns:
        Tuple (prob, d_prob): forward probability and its derivative.
    """
    M = len(x)  # Length of the observation sequence
    alpha = np.zeros((M, K))       # Forward probabilities
    alpha_der = np.zeros((M, K))   # Derivatives of forward probabilities

    # Initialization at position 0 
    for z in range(K):
        e_prob = emission_prob(x[0], z, q, p, 0)  # Emission probability at first position
        if param_index == 'q' and z == q_index:
            # Derivative of emission probability w.r.t. q[q_index]
            e_der = emission_prob_der(x[0], z, q, p, 0, param_index, q_index)
        else:
            e_der = 0.0

        # Standard forward probability
        alpha[0, z] = q[z] * e_prob

        # Derivative of alpha at time 0
        if param_index == 'q' and z == q_index:
            alpha_der[0, z] = q[z] * e_der + e_prob 
        else:
            alpha_der[0, z] = q[z] * e_der

    # Recursion over positions 1 to M-1 
    for i in range(1, M):
        for z in range(K):
            sum_alpha = 0.0      # Sum of previous alphas * transition probabilities
            sum_alpha_der = 0.0  # Derivative of the above sum

            for z_prev in range(K):
                # Transition probability from z_prev to z
                trans = transition_prob(z_prev, z, d_list[i-1], r, q)

                # Derivative of transition probability
                trans_der = transition_prob_der(z_prev, z, d_list[i-1], r, q, param_index, q_index)

                # Contribution to alpha from z_prev
                contrib = alpha[i-1, z_prev] * trans
                contrib_der = alpha_der[i-1, z_prev] * trans + alpha[i-1, z_prev] * trans_der  # Product rule

                sum_alpha += contrib
                sum_alpha_der += contrib_der

            # Emission probability at current position
            e_prob = emission_prob(x[i], z, q, p, i)

            if param_index == 'q' and z == q_index:
                # Derivative of emission probability w.r.t. q[q_index]
                e_der = emission_prob_der(x[i], z, q, p, i, param_index, q_index)
            else:
                e_der = 0.0

            # Forward probability at time i for state z
            alpha[i, z] = sum_alpha * e_prob

            # Derivative of alpha using product rule
            alpha_der[i, z] = sum_alpha_der * e_prob + sum_alpha * e_der

    # Final output: sum over final alphas and their derivatives
    prob = np.sum(alpha[-1])        # Total probability of observed sequence
    d_prob = np.sum(alpha_der[-1])  # Derivative of total probability

    return prob, d_prob


def fisher_information_matrix_diploid(K, q, p, d_list, r):
    n_params = K + 1  # q_0,...,q_{K-1}, r
    FI = np.zeros((n_params-1, n_params-1))
    M = len(p[0])
    # For diploid: sum over all pairs of (x1, x2)
    
    for x1 in itertools.product([0, 1], repeat=M):
        for x2 in itertools.product([0, 1], repeat=M):
            # Joint probability: product of two independent chromosomes
            p_x1 = forward_algorithm(x1, K, q, p, d_list, r)
            p_x2 = forward_algorithm(x2, K, q, p, d_list, r)
            p_joint = p_x1 * p_x2
            grads = []
            # Derivative w.r.t. each q
            for k in range(K-1):
                _, dq1 = forward_algorithm_derivative(x1, K, q, p, d_list, r, param_index='q', q_index=k)
                _, dq2 = forward_algorithm_derivative(x2, K, q, p, d_list, r, param_index='q', q_index=k)
                # Here: we use that the sum of q is equal to 1
                _, dq1_last = forward_algorithm_derivative(x1, K, q, p, d_list, r, param_index='q', q_index=K - 1)
                _, dq2_last = forward_algorithm_derivative(x2, K, q, p, d_list, r, param_index='q', q_index=K - 1)
                # We aim to calculate the derivation of the log-likeliood and not of 
                # the likelihood
                grad_k = (dq1/p_x1  + dq2/p_x2 ) - (dq1_last/p_x1  + dq2_last/p_x2)
                #print(p_x1, p_x2)
                #print(dq1_last, dq2_last, dq1, dq2)
                grads.append(grad_k)
                #grads.append(dq1 + dq2)  # Chain rule: sum of derivatives for both chromosomes
            # Derivative w.r.t. r
            _, dr1 = forward_algorithm_derivative(x1, K, q, p, d_list, r, param_index='r')
            _, dr2 = forward_algorithm_derivative(x2, K, q, p, d_list, r, param_index='r')
            grads.append(dr1/p_x1  + dr2/p_x2)
            grads = np.array(grads)
            FI += p_joint * np.outer(grads, grads)
    return FI 

# Example usage
K = 2
M = 4
q = [0.3, 0.7]
p = [
    [0.2, 0.3, 0.7, 0.1], [0.9, 0.8, 0.6, 0.9]
   
]

r = 0.5
d_list = [1.0, 1.0, 1.0, 1.0]

FI_diploid = fisher_information_matrix_diploid(K, q, p, d_list, r)
print("Fisher information matrix (diploid):")
print(np.linalg.inv(FI_diploid))

