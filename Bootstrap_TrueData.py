# Compare Fisher Information Admixture Model with Boostrap

import numpy as np
import sys
import os
import pandas as pd
#  Statistical Test for Linkage Model vs. Admixture Model
script_dir = os.path.dirname(os.path.abspath(__file__)) 
module_path = os.path.join(script_dir) 
sys.path.append(module_path)

import Simulation_Linkage_diploid as sim
import cM_calculation as cm
#%%
# Structure
# 1: calc x, p, d
# 2: apply bootstrap

# 1)

#  Statistical Test for Linkage Model vs. Admixture Model
script_dir = os.path.dirname(os.path.abspath(__file__)) 
module_path = os.path.join(script_dir) 
sys.path.append(module_path)


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


#%%

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

df_results = cm.create_df()
d_all = cm.get_distances_per_chromosome(df_results)
#%%


# 2)
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
    
    for _ in range(n_iter):
        q = np.random.dirichlet(np.ones(K))
        r = np.random.uniform(0,6)

        current_ll = compute_likelihood_multiple_chromosome_diploid(
            X1, X2, q, p, d_t, r
        )
        if current_ll > best_log_likelihood:
            best_log_likelihood = current_ll
            best_q = q.copy()
            best_r = r
    
    return best_q, best_r, best_log_likelihood


def simulation(numRep, K, q, r, d_values, p):
    res = []
    res_r = []

    for i in range(numRep):
        print(i)
        x2_all = []
        x1_all = []
        counter = 0
        for i_p in p:
            d = d_values[counter]
            M = len(d)
            print(d[0], i_p)
            if(M >= 2 or d[0] > -1):
                x1, x2 = sim.simulate_markov_chain_diploid(K, q, r, M+1, [0] + d, np.array(i_p))
            elif(M == 1):
                c = sum(x * y for x, y in zip(i_p[0], q)) # success probability
                X_sum = np.random.binomial(n=2, p=c)
                if(X_sum == 1):
                    x1 = [0]
                    x2 = [1]
                else:
                    x1 = [int(X_sum/2)]
                    x2 =[int(X_sum/2)]
            counter+= 1
            x1_all.append(x1)
            x2_all.append(x2)
        print(x1_all, x2_all)
        est = random_search(x1_all, x2_all, p, d_values, K, n_iter=1000)
        res.append(est[0])
        res_r.append(est[1])
    return res, res_r
    
# Fix q and p and calculate different values for  x
# 2) Calculate the boostrap variance of the data

K = 5
d_values = d_all
p = p_final
q = [0.8, 0.04, 0.04, 0.08, 0.04]
r = 1.8

var_bootstrap = simulation(100, K, q, r, d_values, p_final_new)

print("variance of r", np.var(var_bootstrap[1]))
matrix = np.vstack((var_bootstrap[0]))

variances = np.var(matrix, axis=0, ddof=1)  

cov_matrix = np.cov(matrix.T, ddof=1)

print("Varianz pro Komponente:", variances)
print("\nKovarianzmatrix:\n", cov_matrix)
