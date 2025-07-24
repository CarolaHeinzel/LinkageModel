import numpy as np
import sys
import os
import pandas as pd
#  Statistical Test for Linkage Model vs. Admixture Model if we consider the whole population 
# It is important that the calculation of the allele frequencies is done with other individuals than the individuals in the population.
script_dir = os.path.dirname(os.path.abspath(__file__)) 
module_path = os.path.join(script_dir) 
sys.path.append(module_path)
import cM_calculation as cm
import Simulation_maximization_Admixture as sA
import Test_LM_AM_diploid as test_py
df_results = cm.create_df()
#%%
# Calculate the MLE and the Allele Frequencies and d
# 1) Read the Data
path = "/home/ch1158/Downloads/1000G_AIMsetKidd(3).vcf"
res_all = []
def read_vcf(path):
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            res_all.append(line)
    return res_all
res_x = read_vcf(path)
res_x_rm = res_x[255:310]
x_all1 = []
x_all2 = []
for i in res_x_rm:
    genotypes = [gt for gt in i.strip().split('\t') if gt]
    left = [item.split('|')[0] for item in genotypes[9:]]
    right = [item.split('|')[1] for item in genotypes[9:]]

    x_all1.append(left) # 55 times 2504 many entries, i.e. [marker, individual]
    x_all2.append(right)

x_all_inv1 = np.array(x_all1).astype(int) #transpose() # phased data [marker, individual]
x_all_inv2 = np.array(x_all2).astype(int) #transpose() # phased  data
x_all_inv1 = x_all_inv1.transpose()
x_all_inv2 = x_all_inv2.transpose()

#%%
path = "/home/ch1158/Downloads/Frequencies_1000G_AIMsetKidd.csv"
data_p = pd.read_csv(path)
# EUR, AFR, SAS, EAS, AMR
continent_map = {
    "EUR": ["GBR", "FIN", "IBS", "TSI", "CEU"],
     "AMR": ["MXL", "CLM", "PUR", "PEL"],
     "SAS": ["GIH", "PJL", "BEB", "STU", "ITU"],
     "EAS": ["KHV", "CDX", "CHB", "JPT", "CHS"],
     "AFR": ["ACB", "GWD", "MSL", "ESN", "YRI", "LWK", "ASW"]
}
df_grouped = pd.DataFrame()
df_grouped['Position'] = data_p['Unnamed: 0']  

for continent, populations in continent_map.items():
    df_grouped[continent] = data_p[populations].mean(axis=1)
p = np.array(df_grouped.iloc[:,1:]).transpose() # allele frequencies


def correct_x(x):
    gruppen = [3, 7, 1, 4, 2, 2, 1, 5, 1, 1, 2, 2, 5, 1, 5, 1, 5, 4, 1, 1, 1]  
    result = []
    start = 0
    for g in gruppen:
        result.append(list(x[start:start+g]))
        start += g
    
    #print(result)
    return result

p_final = correct_x(p.T)
p_final_new = []
for i in p_final:
    res = []
    for j in i:
        res.append(list(j))
    p_final_new.append(res)
#%%
# Also divide the values for X according to the chromosome


res_x = correct_x(x_all_inv1[0])
print(res_x)
print(x_all_inv1[0])
#%%

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

    M = len(p[0])        # number of markers
    K = len(q)        # number of hidden states
    #print("p", p)
    # Initialize forward probabilities alpha
    alpha = np.zeros((M, K))
    # Initial step: alpha_1(k) = P(Z_1 = k) * P(X_1 | Z_1 = k)
    for k in range(K):
     #   print(p[k][0])
        emission_prob = q[k] * p[k][0] if X[0] == 1 else 1 - q[k] * p[k][0]
        alpha[0, k] = q[k] * emission_prob
    # Recursive step
    for m in range(1, M):
        for k in range(K):
            sum_prob = 0.0
            for k_prev in range(K):
                if k == k_prev:
                    trans_prob = np.exp(-d[m-1] * r) + (1 - np.exp(-d[m-1] * r)) * q[k_prev]
                else:
                    trans_prob = (1 - np.exp(-d[m-1] * r)) * q[k_prev]
                sum_prob += alpha[m - 1, k_prev] * trans_prob
            emission_prob = q[k] * p[k][m] if X[m] == 1 else 1 - q[k] * p[k][m]
            alpha[m, k] = sum_prob * emission_prob
    # Total likelihood = sum over final alphas
    return np.sum(alpha[-1])


def compute_likelihood_multiple_chromosome_diploid(X1, X2, q, p, d, r):
    l = 0
    res = 1
    for i in p:
        l1 =  compute_likelihood_single_chromosome(X1[l], q, np.array(i).T, d[l], r)
        l2 =  compute_likelihood_single_chromosome(X2[l], q, np.array(i).T, d[l], r)
        l += 1
        res *= l1 * l2
    return res

def random_search(X1, X2, p, d_t, K, n_iter=1000):

    best_log_likelihood = -np.inf
    best_q = None
    best_r = None
    N = len(X1)
    #print(N)
    for _ in range(n_iter):
        q = np.random.dirichlet(np.ones(K), N)
        r = np.random.uniform(0,6)
        current_ll = 1
        for n in range(N):
           # print(q[n])
           # print(X1[n])
            x1 = correct_x(X1[n])
            x2 = correct_x(X2[n])
            current_ll *= compute_likelihood_multiple_chromosome_diploid(
                x1, x2, q[n], p, d_t, r
                )
        if current_ll > best_log_likelihood:
            best_log_likelihood = current_ll
            best_q = q.copy()
            best_r = r
    
    return best_q, best_r, best_log_likelihood
summ = 0
val_all = []
x1_all = []
x2_all= []
for i in range(0, 10):
    x1 = x_all_inv1[i]
    x2 = x_all_inv2[i]
    x1_all.append(x1)
    x2_all.append(x2)

#X, p, d, r_range, q_range, K, d_t
x1 = correct_x(x_all_inv1[2500])
x2 = correct_x(x_all_inv2[2500])
d_all = cm.get_distances_per_chromosome(df_results)
#res = random_search([x1, x2], [x2,x2],  p_final_new,  d_all, 5)

res = random_search(x1_all, x2_all,  p_final_new,  d_all, 5)
#res = compute_likelihood_multiple_chromosome_diploid(x1, x2, q, p_final, d_all, 1)
print(res)
#%%
# Now: we compare the values in a statistical test!!!
def test(X1_all, X2_all, q_Am, q_LM, p, p_origi, d, r):

    # Linkage Model
    l_LM = 1
    l_AM = 1
    N = len(X1_all)
    for i in range(N):
        X1 = X1_all[i]
        X2 = X2_all[i]
        x1 = correct_x(X1)
        x2 = correct_x(X2)
        logL_unrestricted = (compute_likelihood_multiple_chromosome_diploid(x1, x2, q_LM[i], p, d, r))
        l_LM *= logL_unrestricted
        X = np.array(X1) + np.array(X2)
    # Admixture
        logL_restricted = test_py.likelihood(X, q_Am[i], p_origi.T)
        l_AM *= np.exp(logL_restricted)
    result = test_py.likelihood_ratio_test(logL_restricted[0], logL_unrestricted)
    return(result)

def test_sumary(X1_all, X2_all, p, p_origi, d, K):
    N = len(X1_all)
    q_AM_all = []
    for i in range(N):
        X1 = X1_all[i]
        X2 = X2_all[i]
        X = np.array(X1) + np.array(X2)
    # MLE Admixture Model
        q_AM = sA.get_admixture_proportions(np.array([X]), p_origi)
        q_AM_all.append(q_AM)

    q_LM, r, lik = random_search(X1_all, X2_all, p_final,  d, 5)
    # MLE Admixture
    t = test(X1, X2, q_AM_all, q_LM, p, p_origi, d, r)
    return t 
K = 5
res_test_all = test_sumary(x1_all, x2_all, p_final, p, d_all,  K)
print(res_test_all)
