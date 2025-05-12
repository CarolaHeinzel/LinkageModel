#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 12:12:45 2025

@author: ch1158
"""
# Evaluate the statistical Test

import numpy as np
import sys
sys.path.append("/home/ch1158/Ergebnisse_Paper/HMM/")
import Simulation_Linkage as sL
import Simulation_maximization_Admixture as sA
import Grid_Search_Linkage as grid_l
import Test_LM_AM as test


# Truth is the linkage model
def test_LM(d, r, M , K, q):
     
    # Simulate the data according to the Linkage Model

    # Create the true allele frequencies
    p = sL.create_random_matrix(M, K)

    # Create time-dependent distance values
    d_values = [d] * M  # Example: all distances are the same

    # Simulate the hidden states according to the linkage model
    states_visited, Q = sL.simulate_markov_chain(K, q, r, M, d_values, p)

    # Apply the test
    t = test.test_sumary(states_visited, p, d_values)

    return t
 
def test_AM(M, K, q, d):
    
    p = sL.create_random_matrix(M, K)
    X = test.create_sample_pbekannt(M, K, p.T, q)
    
    d_values = [d]*M
    t = test.test_sumary(X, p, d_values)
    
    return t


# Repeat the test numRep times if the truth is the Admixture Model and 
# if the truth is the linkage model

def rep_test(numRep, d, r, M , K, q):
    
    result_Am = [0] * numRep
    result_Lm = [0]*numRep
    for i in range(numRep):
        t_temp_AM = test_AM(M, K, q, d)
        if(t_temp_AM['reject_H0'] == False):
            result_Am[i] = 1
        t_temp_LM = test_LM(d, r, M , K, q)
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



try_test = rep_test_d(100, [0.1, 0.2, 0.5, 1, 5], 1, 100, 2, np.array([0.2, 0.8]))
#%%

import matplotlib.pyplot as plt
import numpy as np

# Daten
x = [0.1, 0.2, 0.5, 1, 5]  # x-Werte
y = try_test  # y-Werte: zwei Listen nebeneinander für jeden x-Wert
colors = ['#1f77b4', 'green']  # z.B. Blau und Orange
# Anzahl der Gruppen (hier 2: für [2,1] und [2,1])
num_groups = len(y)

# Balken-Breite
width = 0.35

# Setze Positionen für die Balken nebeneinander
x_pos = np.arange(len(x)) # Use np.arange for better spacing

# Erstelle Balken
fig, ax = plt.subplots()
manual_labels = ['Type 1 Error', 'Type 2 Error']
for i in range(num_groups):
    print(y[i])
    liste = [1-y[i][j]/100 for j in range(0, 5)]
    # Calculate the offset for each group
    offset = (i - (num_groups - 1) / 2) * width
    # Plot the bars for the current group
    ax.bar(x_pos + offset, liste, width, label=manual_labels[i],  color=colors[i] ) 
    # y[i] directly accesses the data for the current group

# Labels und Titel
ax.set_xlabel('d')
ax.set_title('Evaluation of the Test for r = 1, M = 100')
ax.set_xticks(x_pos)
ax.set_xticklabels(x) 
ax.legend()

plt.show()
