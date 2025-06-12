import numpy as np
import math

# Calculates the Fischer Information for one Chromosome

# 1) Calculates the derivations
def forward_algorithm_derivative(x, K, q, p, d_list, r, 
                                 transition_prob, transition_prob_der,
                                 emission_prob, emission_prob_der,
                                 param_index='q', q_index=0):
    M = len(x)
    
    alpha = np.zeros((M, K))
    alpha_der = np.zeros((M, K))
    
    # Initialisierung i = 0
    for z in range(K):
        e_prob = emission_prob(x[0], z, q, p, 0)
        e_der  = emission_prob_der(x[0], z, q, p, 0)
        
        alpha[0, z] = e_prob
        if param_index == 'q' and z == q_index:
            alpha_der[0, z] = e_der
        else:
            alpha_der[0, z] = 0

    # Rekursion i >= 1
    for i in range(1, M):
        for z in range(K):  # aktueller Zustand z
            sum_alpha = 0
            sum_alpha_der = 0
            for z_prev in range(K):
                trans = transition_prob(z_prev, z, d_list[i-1], r, q)
                if(z == q_index):
                    trans_der = transition_prob_der(z_prev, z, d_list[i-1], r, q, param_index) 
                else:
                    trans_der = 0
                contrib = alpha[i-1, z_prev] * trans
                contrib_der = (
                    alpha_der[i-1, z_prev] * trans +
                    alpha[i-1, z_prev] * trans_der
                )
                sum_alpha += contrib
                sum_alpha_der += contrib_der

            e_prob = emission_prob(x[i], z, q, p, i)
            e_der  = emission_prob_der(x[i], z, q, p, i)

            alpha[i, z] = sum_alpha * e_prob
            alpha_der[i, z] = sum_alpha_der * e_prob + sum_alpha * e_der

    # Gesamtwahrscheinlichkeit und Ableitung
    prob = np.sum(alpha[-1])
    d_prob = np.sum(alpha_der[-1])

    return prob, d_prob



def emission_prob(x_i, z, q, p, i):
    """P(X_i | Z_i = z)"""
    q_z = q[z]
    p_zi = p[z][i]
    return  ((q_z *p_zi)**x_i) * ((1 - q_z *p_zi)**(1 - x_i))




def transition_prob_der(z_prev, z_next, d, r, q, param_index):
    if(param_index == "r"):
        """Compute P(Z_{i+1} = z_next | Z_i = z_prev)"""
        if z_prev == z_next:
            return math.exp(-r * d)*(-d) + d* math.exp(-r * d) * q[z_next]
        else:
            return (d* math.exp(-r * d)) * q[z_next]
    # 0 if derivative not with respect to z_next!
    else:
        K = len(q)
        if z_next < K:
            return  (1 - math.exp(-r * d)) 
        else:
            return -(1 - math.exp(-r * d))
  # derivative of emission probability  
def emission_prob_der(x_i, z, q, p, i):
    """P(X_i | Z_i = z)"""
    p_zi = p[z][i]
    res = p_zi # x_i = 1
    if(x_i == 0):
        res = - p_zi
    return res



def transition_prob(z_prev, z_next, d, r, q):
    """Compute P(Z_{i+1} = z_next | Z_i = z_prev)"""
    if z_prev == z_next:
        return math.exp(-r * d) + (1 - math.exp(-r * d)) * q[z_next]
    else:
        return (1 - math.exp(-r * d)) * q[z_next]
    


K = 2                     # number of hidden states
M = 4                     # length of observation
x = [1, 0, 1, 1]          # example observed sequence
q = [0.3, 0.7]            # q_z for z = 0, 1
r = 0.5                   # transition decay rate
p = [                    # p[z][i] = p_{z,i}
    [0.2, 0.3, 0.7, 0.8],  # for z = 0
    [0.9, 0.8, 0.6, 0.4]   # for z = 1
]
d_list = [1.0, 1.0, 1.0]  # time differences

# derivative with respect to r
prob, d_prob_r = forward_algorithm_derivative(
    x, K, q, p, d_list, r,
    transition_prob, transition_prob_der,
    emission_prob, emission_prob_der,
    param_index='r'
)

# derivative with respect to q, population is clear by q_index
prob, d_prob_q0 = forward_algorithm_derivative(
    x, K, q, p, d_list, r,
    transition_prob, transition_prob_der,
    emission_prob, emission_prob_der,
    param_index='q', q_index=0
)


#%%
# 2 Caclulates the probability of  X_1,..., X_M

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

#%%
# Combines everything to get the Fischer Information
import itertools

def fischer_information( K, q, p, d_list, r):
    e_all = 0
    kombinationen = list(itertools.product([0, 1], repeat=M))
    for x in kombinationen:
        p_temp = forward_algorithm(x, K, q, p, d_list, r)
        temp, d_temp_r = forward_algorithm_derivative( x, K, q, p, d_list, r, transition_prob, transition_prob_der, emission_prob, emission_prob_der, param_index='r')
        # All derivations
        q_all = [d_temp_r]
        for k in range(K-1):
            temp, d_temp_q = forward_algorithm_derivative( x, K, q, p, d_list, r, transition_prob, transition_prob_der, emission_prob, emission_prob_der, param_index='q', q_index = k)
            q_all.append(d_temp_q)
        print(q_all)
        q_all = np.array(q_all)
        e_all += p_temp * np.outer(q_all, q_all)
        
    return e_all
K = 2                     # number of hidden states
M = 4                     # length of observation
#x = [1, 0, 1, 1]          # example observed sequence
q = [0.3, 0.7]            # q_z for z = 0, 1
r = 0.5                   # transition decay rate
p = [                    # p[z][i] = p_{z,i}
    [0.2, 0.3, 0.7, 0.1],  # for z = 0
    [0.9, 0.8, 0.6, 0.9]   # for z = 1
]
d_list = [1.0, 1.0, 1.0]  # time differences
test = fischer_information(K, q, p, d_list, r)
print(test)
#%%
print(np.linalg.inv(test))
