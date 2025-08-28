#!/usr/bin/env python3
# Evaluate the statistical Test: this is the main document to evaluate the test
import numpy as np
import sys
import random
sys.path.append("/home/ch1158/Ergebnisse_Paper/HMM/")
import Simulation_Linkage_diploid as sL # Simulates data according to the linkage model
import Test_LM_AM_diploid as test # Implements the statistical test

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

# Truth is the linkage model
def test_LM(d, r, M , K, q):
     
    # Simulate the data according to the Linkage Model
    
    # Create the true allele frequencies
    p = sL.create_random_matrix(M, K)

    # Create time-dependent distance values
    d_values = [d] * M  # Example: all distances are the same

    # Simulate the hidden states according to the linkage model
    states_visited1, states_visited2 = sL.simulate_markov_chain_diploid(K, q, r, M, d_values, p)

    # Apply the test
    t = test.test_sumary(states_visited1, states_visited2, p, d_values, K)

    return t
 
def test_AM(M, K, q, d):
    
    p = sL.create_random_matrix(M, K)
    X = test.create_sample_pbekannt(M, K, p.T, q)
    
    d_values = [d]*M

    X1, X2 = split_and_halve(X)
    t = test.test_sumary(X1, X2, p, d_values, K)
    
    return t


# Repeat the test numRep times if the truth is the Admixture Model and 
# if the truth is the linkage model

def rep_test(numRep, d, r, M , K, q):
    
    result_Am = [0] * numRep
    result_Lm = [0]*numRep
    for i in range(numRep):
        print(i)
        t_temp_AM = test_AM(M, K, q, d)
        print("a", t_temp_AM)
        if(t_temp_AM['reject_H0'] == False):
            result_Am[i] = 1
        
        t_temp_LM = test_LM(d, r, M , K, q)
        print("b", t_temp_LM)
        if(t_temp_LM['reject_H0'] == True):
            result_Lm[i] = 1
    
    return result_Am, result_Lm

# repeat it for different values of d


def rep_test_d(numRep, d_list, r, M , K, q):
    result_Am = [0] * len(d_list)
    result_Lm = [0]*  len(d_list)
    ind = 0
    for d in  d_list:
        AM, LM = rep_test(numRep, d, r, M , K, q)
        result_Am[ind] = sum(AM)
        result_Lm[ind] = sum(LM)
        ind += 1
    
    return result_Am, result_Lm

K = 2
try_test_r100 = rep_test_d(100, [0.1, 0.5,  1, 2, 5, 10], 1, 100, K, np.array([0.2, 0.8]))


