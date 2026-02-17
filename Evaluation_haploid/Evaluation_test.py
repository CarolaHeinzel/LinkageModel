# Evaluate the statistical Test

import numpy as np


import Simulation_Linkage as sL
import Test_LM_AM as test

import matplotlib.pyplot as plt


# Truth is the linkage model
def test_LM(d, r, M, K, q):
    p = sL.create_random_matrix(M, K)              
    d_values = np.full(M, float(d), dtype=float)

    X, _Q = sL.simulate_markov_chain(K, q, r, M, d_values, p)
    t = test.test_summary(X, p, d_values, K, q, r)
    return t


def test_AM(M, K, q, d, r):
    p = sL.create_random_matrix(M, K)
    X = test.create_sample_pbekannt(M, K, p, q)

    d_values = np.full(M, float(d), dtype=float)
    t = test.test_summary(X, p, d_values, K, q, r)
    return t


def rep_test(numRep, d, r, M, K, q):
    # AM: count "do not reject"
    # LM: count "reject" (power)
    accept_AM = 0
    reject_LM = 0

    for i in range(numRep):
        print("i", i)
        t_AM = test_AM(M, K, q, d, r)
        if t_AM["reject_H0"] is False:
            accept_AM += 1 # AM is truth
        print(accept_AM)
        t_LM = test_LM(d, r, M, K, q) # LM is truth
        if t_LM["reject_H0"] is True:
            reject_LM += 1
        print(reject_LM)

    return accept_AM / numRep, reject_LM / numRep


def rep_test_d(numRep, d_list, r, M, K, q):
    accept_AM_rates = []
    reject_LM_rates = []

    for d in d_list:
        acc, powr = rep_test(numRep, d, r, M, K, q)
        accept_AM_rates.append(acc)
        reject_LM_rates.append(powr)

    return np.array(accept_AM_rates), np.array(reject_LM_rates)


# ---- run ----
numRep = 100
d_list = [0.1, 1, 2, 5]
r = 0.1
M = 10000
q = np.array([0.2, 0.8], dtype=float)
K = len(q)

a_AM = []
a_LM = []

for M in [1000]:
    for r in [0.1, 1, 10]:
        accept_AM, reject_LM = rep_test_d(numRep, d_list, r, M, K, q)
        print("AM truth accept rates:", accept_AM)
        print("LM truth reject rates (power):", reject_LM)
        a_AM.append(accept_AM)
        a_LM.append(reject_LM)
