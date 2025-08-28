# Implements a statistical test to compare the Linkage Model to the Admixture Model


# To apply the test to some real data, we just have to change the likelihood 
# for the Hidden Markov Model
import numpy as np
from scipy.stats import chi2
from scipy.stats import binom
import math
import sys
import random
sys.path.append("/home/ch1158/Ergebnisse_Paper/HMM/")
# Load the other data
import Simulation_maximization_Admixture as sA
import Grid_Search_Linkage_diploid as grid_l

def split_and_halve(input_list):
    list1 = []
    list2 = []
    for num in input_list:
        if num == 1:
            u =  random.uniform(0, 1)
            if(u < 0.5):
                list1.append(num)
                list2.append(0)
            else:
                list2.append(num)
                list1.append(0)   
        else:
            half = num / 2
            list1.append(half)
            list2.append(half)
    return list1, list2

def likelihood(x, q, p):
    l = 0
    M = len(x)
    for m in range(M):
        theta = np.dot(q, p[m])
        theta = np.clip(theta, 1e-5, 1 - 1e-5)  # avoid log(0)
        
        # Binomial log-likelihood
        log_binom_coeff = 0 #np.log(comb(2, x[m]))
        l += log_binom_coeff + x[m] * np.log(theta) + (2 - x[m]) * np.log(1 - theta)
    return l

# Simulates according to the Admixture Model
def create_sample_pbekannt(M,  K, p, q):
    x = np.zeros(M)
    loc = np.dot(q, p)
    #print(loc)
    for m in range(M):
        x[m] = binom.rvs(n=2, p=loc[m])
    return x 
    
# Calculate the likelihood of the Linkage Model, given the MLEs
# Forward Algorithm 
# Explained in Appendix A.3.1
def emission_prob(x_i, z, q, p, i):
    """Compute P(X_i | Z_i = z)"""
    q_z = q[z]
    p_zi = p[i][z]
    return  ((q_z *p_zi)**x_i) * ((1 - q_z*p_zi)**(1 - x_i))

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

def forward_algorithm_diploid(X1, X2, K, q, p, d_list, r):
    l1 = forward_algorithm(X1, K, q, p, d_list, r)
    l2 = forward_algorithm(X2, K, q, p, d_list, r)
    return l1 * l2


def likelihood_ratio_test(logL_restricted, logL_unrestricted, df = 1, alpha=0.05):
    """
    Perform a Likelihood Ratio Test (LRT).

    Parameters:
    logL_restricted (float): Log-likelihood of the restricted model (H0)
    logL_unrestricted (float): Log-likelihood of the unrestricted model (H1)
    df (int): Degrees of freedom (number of restrictions)
    alpha (float): Significance level (default 0.05)

    Returns:
    dict: Contains LR statistic, p-value, critical value, and decision
    """
    # Calculate the LR statistic
    LR_stat = -2 * (logL_restricted - logL_unrestricted)

    # Compute p-value
    p_value = 1 - chi2.cdf(LR_stat, df)

    # Critical value from chi^2 distribution
    critical_value = chi2.ppf(1 - alpha, df)

    # Decision
    reject_H0 = LR_stat > critical_value

    return {
        "LR_statistic": LR_stat,
        "p_value": p_value,
        "critical_value": critical_value,
        "reject_H0": reject_H0
    }


logL_restricted = -520.5
logL_unrestricted = -115.0
df = 1  # z.B. zwei Einschränkungen

result = likelihood_ratio_test(logL_restricted, logL_unrestricted, df)
print(result)
#%%

 
# test
def test(X1, X2, q_Am, q_LM, p, d, r):
    # Linkage
    # Attention: only works for K = 2
    logL_unrestricted = np.log(forward_algorithm_diploid(X1, X2, K, [q_LM, 1-q_LM], p, d, r))
    # Admixture
    X = np.array(X1) + np.array(X2)

    logL_restricted = likelihood(X, q_Am, p)
    print("l", logL_restricted[0], logL_unrestricted)
    result = likelihood_ratio_test(logL_restricted[0], logL_unrestricted)
    return(result)



def test_sumary(X1, X2, p, d, K):
    # MLE Linkage 
    print("a")
    q_LM, r, lik = grid_l.grid_search(X1, X2, p.T,  np.linspace(0,100, 500), K, d)
    # MLE Admixture
    print("c", q_LM, r, lik)
    X = np.array(X1) + np.array(X2)
    print(len(X), X)
    q_AM = sA.get_admixture_proportions(np.array([X]), p.T)
    print(q_AM)
    t = test(X1, X2, q_AM, q_LM, p, d, r)
    return(t)

M  = 100
K = 2
d_values = [0.1]*M  # Example time-dependent values for d_t
#p = sL.create_random_matrix(M, K)
#X = create_sample_pbekannt(M, K, np.array(p).T, [0.2, 0.8] )
#X1, X2 = split_and_halve(X)
#q = [0.1, 0.9]
#X1, X2 =  sL.simulate_markov_chain_diploid(K, q, 1, M, d_values, p)
#res_AM = test_sumary(X1, X2, np.array(p), d_values,  K)
#print(res_AM)
#%%
#q = [0.2, 0.8]
#for i in range(100):
#    X1, X2 =  sL.simulate_markov_chain_diploid(K, q, 10, M, d_values, p)
    #X = create_sample_pbekannt(M, K, np.array(p).T, [0.2, 0.8] )
    #X1, X2 = split_and_halve(X)
    #print(X1)
#    res_AM = test_sumary(X1, X2, np.array(p), d_values,  K)
#    print(res_AM)
#%%
#d_values = [0.5] * M
#r = 0.2
#X = sL.simulate_markov_chain(K, q, r, M, d_values, p)
#res_AM = test_sumary(X[0], np.array(p), d_values,  K)
#print(res_AM)
