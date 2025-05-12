# Implements a statistical test to compare the Linkage Model to the Admixture Model

import numpy as np
from scipy.stats import chi2
from scipy.stats import dirichlet, binom


# Calculate the likelihood of the Admixture Model given the MLEs
def likelihood(x, q, p):
     l = 0
     M = len(p)
     N = len(q)
     for i in range(N):
         for m in range(M):
             theta = np.dot(q.iloc[i,:].tolist(), p[m, :])
             l += x[m][i] * np.log(1-theta) + (2-x[m][i])*np.log(theta)
             
     return l
 
# Calculate the likelihood of the Linkage Model, given the MLEs
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


X = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 1])

# Anzahl der versteckten Zustände
K = 2

# Initialverteilung q_k über Z_1 (z. B. gleichverteilt)
q = np.array([0.5, 0.5])

# p_{k,m} – Bernoulli-Wahrscheinlichkeiten für jeden Zustand k an jeder Position m
# p[k, m] = Erfolgswahrscheinlichkeit im Zustand k bei Marker m
p = np.array([
    [0.9, 0.2, 0.8, 0.85, 0.1, 0.9, 0.2, 0.8, 0.85, 0.1],  # Zustand 1
    [0.6, 0.5, 0.3, 0.4, 0.7, 0.6, 0.5, 0.3, 0.4, 0.7]    # Zustand 2
])

# Distanzen d_m (z. B. genetische Distanz in cM) – d[0] ignorieren wir, also Dummy
d = np.array([0.0, 0.1, 0.1, 0.1, 0.1,0.1, 0.1, 0.1, 0.1, 0.1])

# Rekombinationsrate
r = 1

# Likelihood berechnen
likelihood = compute_likelihood_single_chromosome(X, q, p, d, r)

import matplotlib.pyplot as plt

# q-Werte in [0, 1]
q_values = np.linspace(0, 1, 100)

# Berechnung des Outputs für alle q-Werte
likelihood_values = [compute_likelihood_single_chromosome(X, np.array([q,1-q]), p, d, r) for q in q_values]

# Plotten
plt.plot(q_values, likelihood_values, label='Likelihood')
plt.xlabel('q')
plt.ylabel('Likelihood')
plt.title('Likelihood vs q')
plt.grid(True)
plt.legend()
plt.show()
#%%
def compute_likelihood_LM(X, q, p, d, r):
    l = 1
    for i in range(len(p)): # Number of Chromosomes
    # Calculate the likelihood for every chromosome 
        l_temp = compute_likelihood_single_chromosome(X[i], q, p[i], d[i], r)
        l *= l_temp
    return l


likelihood = compute_likelihood_LM([X,X], q, [p,p], [d,d], r)



#%%

def likelihood_ratio_test(logL_restricted, logL_unrestricted, df, alpha=0.05):
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


logL_restricted = -120.5
logL_unrestricted = -115.0
df = 2  # z.B. zwei Einschränkungen

result = likelihood_ratio_test(logL_restricted, logL_unrestricted, df)
print(result)
#%%
import sys
sys.path.append("/home/ch1158/Ergebnisse_Paper/HMM/")
import Simulation_Linkage as sL
#%%
# Load the other data
import Simulation_maximization_Admixture as sA
import Grid_Search_Linkage as grid_l


# Data from Linkage Model

M = 100 # Number of rows
K = 2  # Number of states
# Create the true allele frequencies
p = sL.create_random_matrix(M, K)
print(p)
# Parameters
q = np.array([0.2, 0.8])  # Initial probabilities for the first K-1 states
r = 0.5  # Parameter influencing transition probabilities
d_values = [0.1]*M  # Example time-dependent values for d_t
# Simulate the time-dependent Markov chain
states_visited, Q = sL.simulate_markov_chain(K, q, r, M, d_values, p)


#%%

def likelihood(x, q, p):
     l = 0
     M = len(p)

     for m in range(M):
         theta = np.dot(q, p[m, :])  # Weighted average for position m
         l += x[m] * np.log(theta) + (1 - x[m]) * np.log(theta)
     return l
 
# test
def test(X, q_Am, q_LM, p, d, r):
    # Linkage
    logL_unrestricted = np.log(compute_likelihood_single_chromosome(np.array(X), [q_LM, 1-q_LM], p.T, d, r))
    # Admixture
    logL_restricted = likelihood(X, q_Am, p)
    print("l", logL_restricted[0], logL_unrestricted)
    result = likelihood_ratio_test(logL_restricted[0], logL_unrestricted, 1)
    return(result)



def test_sumary(X, p, d):
    # MLE Linkage 
    q_LM, r, lik = grid_l.grid_search(states_visited, p.T,  np.linspace(0,2, 100), K, d_values)
    # MLE Admixture
    q_AM = sA.get_admixture_proportions(np.array([states_visited]), p.T)
    
    t = test(X, q_AM, q_LM, p, d, r)
    print("test", q_LM, q_AM, r)

    return(t)
res = test_sumary(states_visited, p, d_values)
print(res)
# l -289.3921778285122 -98.93548813559582
#%%

# Simulate according to the Linkage Model

def create_sample_pbekannt(M,  K, p, q):
    x = np.zeros(M)

    loc = np.dot(q, p)
    #print(loc)
    for m in range(M):
        #print(x[m])
        #print( binom.rvs(n=2, p=loc[m]))
        x[m] = binom.rvs(n=1, p=loc[m])
    return x


X = create_sample_pbekannt(M, K, p.T, [0.2, 0.8] )
d_values = [10]*M  # Example time-dependent values for d_t

res_AM = test_sumary(X, p, d_values)
print(res_AM)