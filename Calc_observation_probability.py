import math
# Forward Algorithm 
# Explained in Appendix A.3.1
def emission_prob(x_i, z, q, p, i):
    """Compute P(X_i | Z_i = z)"""
    q_z = q[z]
    p_zi = p[z][i]
    return q_z * (p_zi**x_i) * ((1 - p_zi)**(1 - x_i))

def transition_prob(z_prev, z_next, d, r, q):
    """Compute P(Z_{i+1} = z_next | Z_i = z_prev)"""
    if z_prev == z_next:
        return math.exp(-r * d) + (1 - math.exp(-r * d)) * q[z_next]
    else:
        return (1 - math.exp(-r * d)) * q[z_next]

def forward_algorithm(observations, K, q, p, d_list, r):
    """
    observations: list of binary values [X_1, ..., X_M]
    K: number of hidden states
    q: list of q_z, length K
    p: list of lists p[z][i] for p_{z,i}
    d_list: list of d_i between time points
    r: rate parameter
    """
    M = len(observations)
    alpha = [ [0.0 for _ in range(K)] for _ in range(M) ]

    # Initialization (i = 0 corresponds to X_1)
    for z in range(K):
        alpha[0][z] = 1/K * emission_prob(observations[0], z, q, p, 0)

    # Recursion
    for i in range(1, M):
        x_i = observations[i]
        d = d_list[i - 1]  
        for z_next in range(K):
            emission = emission_prob(x_i, z_next, q, p, i)
            total = 0.0
            for z_prev in range(K):
                trans = transition_prob(z_prev, z_next, d, r, q)
                # alpha_i-1 *trans
                total += alpha[i - 1][z_prev] * trans
            # emission does not depend on z_prev, which is the reason why we can leave it here!    
            alpha[i][z_next] = emission * total

    # Termination
    return sum(alpha[M - 1][z] for z in range(K))


# Setup
observations = [1, 0, 1]     # X_1 to X_M
K = 3                        # Number of hidden states

q = [0.9, 0.6, 0.3]          # q_z for each hidden state z = 0,1,2

p = [                        # p[z][i] = p_{z, i+1}
    [0.8, 0.7, 0.6],         # for z = 0
    [0.5, 0.4, 0.3],         # for z = 1
    [0.2, 0.2, 0.2]          # for z = 2
]

d_list = [1.0, 1.5]          # d_1 and d_2
r = 0.5                      # Rate


# Computation
P = forward_algorithm(observations, K, q, p, d_list, r)
print("P(X_1,...,X_M) =", P)

