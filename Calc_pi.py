import math

def emission_prob(x_i, z, q, p, i):
    """P(X_i | Z_i = z)"""
    q_z = q[z]
    p_zi = p[z][i]
    return q_z * (p_zi**x_i) * ((1 - p_zi)**(1 - x_i))

def transition_prob(z_prev, z_next, d, r, q):
    """P(Z_{i+1} = z_next | Z_i = z_prev)"""
    if z_prev == z_next:
        return math.exp(-r * d) + (1 - math.exp(-r * d)) * q[z_next]
    else:
        return (1 - math.exp(-r * d)) * q[z_next]

def filter_predict(k, observations, K, q, p, d_list, r, initial_prob):
    """
    Computes P(Z_k | X_1, ..., X_{k-1}) as normalized distribution.
    
    k: integer (time step ≥ 2)
    Returns: list of size K with probabilities summing to 1.
    """

    assert 1 < k <= len(observations) + 1, "k must be in 2..M+1"

    # Step 1: Compute alpha_{k-1}(z)
    alpha = [ [0.0 for _ in range(K)] for _ in range(k - 1) ]

    # Initialisierung
    for z in range(K):
        alpha[0][z] = initial_prob[z] * emission_prob(observations[0], z, q, p, 0)

    for i in range(1, k - 1):
        d = d_list[i - 1]
        for z_next in range(K):
            total = 0.0
            for z_prev in range(K):
                trans = transition_prob(z_prev, z_next, d, r, q)
                total += alpha[i - 1][z_prev] * trans
            alpha[i][z_next] = emission_prob(observations[i], z_next, q, p, i) * total

    # Step 2: Compute predictive P(Z_k | X_{1:k-1}) via marginalization
    d_prev = d_list[k - 2]  # d_{k-1}
    pred = [0.0 for _ in range(K)]

    for z in range(K):
        total = 0.0
        for z_prev in range(K):
            trans = transition_prob(z_prev, z, d_prev, r, q)
            total += alpha[k - 2][z_prev] * trans
        pred[z] = total

    # Step 3: Normalize
    Z = sum(pred)
    
    # Calculates the probability for each population 
    return [p / Z for p in pred]


observations = [1, 0, 1]
K = 3
q = [0.9, 0.6, 0.3]
p = [
    [0.8, 0.7, 0.6],
    [0.5, 0.4, 0.3],
    [0.2, 0.2, 0.2]
]
d_list = [1.0, 1.5]
r = 0.5
initial_prob = [1/3, 1/3, 1/3]

# Berechne P(Z_3 | X_1, X_2)
k = 3
pred_dist = filter_predict(k, observations, K, q, p, d_list, r, initial_prob)
print(f"P(Z_{k} | X_1,...,X_{k-1}) = {pred_dist}")
