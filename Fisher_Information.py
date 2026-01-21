import numpy as np
import math
def forward_algorithm_derivative(x, K, q, p, d_list, r, 
                                 transition_prob, transition_prob_der,
                                 emission_prob, emission_prob_der,
                                 param_index='q', q_index=0):
    M = len(x)
    
    alpha = np.zeros((M, K))
    alpha_der = np.zeros((M, K))
    
    for z in range(K):
        e_prob = emission_prob(x[0], z, q, p, 0)
        e_der  = emission_prob_der(x[0], z, q, p, 0)
        
        alpha[0, z] = e_prob
        if param_index == 'q' and z == q_index:
            alpha_der[0, z] = e_der
        else:
            alpha_der[0, z] = 0

    for i in range(1, M):
        for z in range(K):  
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
        if z_prev == z_next:
            return  (1 - math.exp(-r * d)) 
        else:
            return (1 - math.exp(-r * d))
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



