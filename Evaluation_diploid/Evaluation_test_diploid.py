# Evaluate the statistical Test (diploid case)

import numpy as np
import Simulation_Linkage_dip as sL
import Test_LM_AM_diploid as test


# Truth is the linkage model (diploid)
def test_LM_diploid(d, r, M, K, q, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    p = sL.create_random_matrix(M, K, rng=rng)
    d_values = np.full(M, float(d), dtype=float)

    X, _Q = sL.simulate_markov_chain_diploid(K, q, r, M, d_values, p, rng=rng)
    t = test.test_summary(X, p, d_values, K, q, r, ploidy=2)
    return t


# Truth is the admixture model (diploid)
def test_AM_diploid(M, K, q, d, r, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    p = sL.create_random_matrix(M, K, rng=rng)
    X = test.create_sample_pbekannt_diploid(M, K, p, q, rng=rng)

    d_values = np.full(M, float(d), dtype=float)
    t = test.test_summary(X, p, d_values, K, q, r, ploidy=2)
    return t


def rep_test_diploid(numRep, d, r, M, K, q, seed=0):
    """Return (type-I accept rate under AM truth, power under LM truth)."""
    rng = np.random.default_rng(seed)

    accept_AM = 0
    reject_LM = 0

    for i in range(numRep):
        print("i", i)

        t_AM = test_AM_diploid(M, K, q, d, r, rng=rng)
        if t_AM["reject_H0"] is False:
            accept_AM += 1

        t_LM = test_LM_diploid(d, r, M, K, q, rng=rng)
        if t_LM["reject_H0"] is True:
            reject_LM += 1

        print("accept_AM", accept_AM, "reject_LM", reject_LM)

    return accept_AM / numRep, reject_LM / numRep


def rep_test_d_list_diploid(numRep, d_list, r, M, K, q, seed=0):
    accept_AM_rates = []
    reject_LM_rates = []

    for d in d_list:
        acc, powr = rep_test_diploid(numRep, d, r, M, K, q, seed=seed)
        accept_AM_rates.append(acc)
        reject_LM_rates.append(powr)

    return np.array(accept_AM_rates), np.array(reject_LM_rates)


# ---- run ----
if __name__ == "__main__":
    numRep = 100
    d_list = [0.1, 1, 2, 5]

    q = np.array([0.2, 0.8], dtype=float)
    K = len(q)

    for M in [50, 100, 200, 500, 1000]:
        for r in [0.1, 1, 10]:
            accept_AM, reject_LM = rep_test_d_list_diploid(numRep, d_list, r, M, K, q, seed=0)
            print("M=", M, "r=", r)
            print("AM truth accept rates:", accept_AM)
            print("LM truth reject rates (power):", reject_LM)
