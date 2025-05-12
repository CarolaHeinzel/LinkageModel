from scipy.optimize import minimize
import numpy as np
from scipy.stats import dirichlet, binom
from scipy.stats import norm
#%%
# Estimator for the IAs
def get_admixture_proportions(x, p, tol=1e-6):
    K, M = p.shape
    res = dirichlet.rvs(alpha=np.ones(K))
    err = 1
    while err > tol:
        loc = fun2(res, p, x)
        err = np.sum(np.abs(res - loc))
        res = loc
    return res
# Function to help get_admixture_proportions
def fun2(q, p, loc_x):
    K, M = p.shape
    E = np.zeros((K, M))
    loc = np.dot(q, p)
    loc[loc==0] = 1e-16
    loc[loc==1] = 1-1e-16
    for k in range(K):
        E[k, :] = (loc_x * p[k, :] / loc + (1 - loc_x) * (1 - p[k, :]) / (1 - loc))
    res = np.sum(E, axis=1) / M * q / 2
    return res / np.sum(res)

# Create Data
def create_sample_pbekannt(M,  K, p, q):
    x = np.zeros(M)

    loc = np.dot(q, p)
    #print(loc)
    for m in range(M):
        #print(x[m])
        #print( binom.rvs(n=2, p=loc[m]))
        x[m] = binom.rvs(n=1, p=loc[m])
    return x
# Simulate Allele Frequencies
print(create_sample_pbekannt(10,  K, p, [0.2, 0.8, 0]))
#%%
def create_p(M, K):
    return np.random.uniform(0, 1, size=(K, M))

p = create_p(10,3)
res = create_sample_pbekannt(10, 1, 3, p, 0.5, 0)
print(res)


#%%


 
#%%
M = 100
K = 2
N = 1
p = create_p(M, K)
x = create_sample_pbekannt(M,  K, p, [0.2, 0.8])
print(get_admixture_proportions(x, p))
