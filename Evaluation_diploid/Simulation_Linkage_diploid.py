import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
# Simulation according to a Hidden Markov Model (based on the Haldane Model)
# diploid individuals

def create_transition_matrix(K, q, r, d_t):
    """
    Create a transition matrix based on the initial probability vector and parameter r over time.

    Parameters:
    K (int): Number of states.
    q (numpy array): Initial probability vector of size K-1.
    q_K (float): Probability for the K-th state.
    r (float): Parameter influencing transition probabilities.
    d_t (float): Time-dependent parameter influencing transition probabilities.
    
    Returns:
    numpy array: A K x K transition matrix.
    """
    Q = np.zeros((K, K))

    # Define diagonal entries
    for k in range(K):
        Q[k, k] = sp.exp(-d_t * r) + (1 - sp.exp(-d_t * r)) * q[k]

    # Define non-diagonal entries
    for i in range(K):
        for j in range(K):
            if i != j:
                Q[i, j] = q[j] *(1 - sp.exp(-d_t * r)) 
    return Q

#d_t = 0
#Q = create_transition_matrix(K, q, r, d_t)
#print(Q)

def step(current_state1, K, q, r, d_t, p):
    """
    Perform one step of the Markov chain and update the current state based on the time-dependent transition matrix.

    Parameters:
    current_state (int): Current state of the Markov chain.
    K (int): Number of states.
    q (numpy array): Initial probabilities for the first K-1 states.
    r (float): Parameter influencing transition probabilities.
    d_t (float): Current time to adjust transition probabilities.

    Returns:
    int: New current state after transition.
    """
    transition_matrix = create_transition_matrix(K, q, r, d_t)
    # Hidden State
    #print(transition_matrix)
    #print(np.array(transition_matrix[current_state1]))
    step1 = np.random.choice(range(K), p=transition_matrix[current_state1])
    # True IAs
    theta1 = q[step1]*p[step1]

    x1 = np.random.binomial(1, theta1)
    return transition_matrix, step1, x1

def simulate_markov_chain(K, q, r, steps, d_values, p):
    """
    Simulate the Markov chain for a given number of steps.

    Parameters:
    K (int): Number of states.
    q (numpy array): Initial probabilities for the first K-1 states.
    r (float): Parameter influencing transition probabilities.
    steps (int): The number of steps to simulate.
    d_values (list): List of time-dependent parameters for each step.
    
    Returns:
    numpy array: Array of states visited.
    """
    pi = q 
    step1 =  np.random.choice(range(K),p = pi)  
    x0 = np.random.binomial(1,p[0,step1] * pi[step1])
    states = [int(x0)]
    for t in range(1, steps):
        Q, step1, x1 = step(step1, K, q, r, d_values[t], p[t])
        #print(current_state)
        states.append(x1)
    
    return states, Q

def create_random_matrix(M, k):
    """
    Create a matrix with M rows and k entries, filled with random values between 0 and 1.

    Parameters:
    M (int): Number of rows.
    k (int): Number of entries in each row.

    Returns:
    numpy.ndarray: A matrix with shape (M, k), filled with random values.
    """
    # Create a matrix filled with random values between 0 and 1
    sharpness = 0.3
    matrix = np.random.dirichlet(np.ones(K) * sharpness, size=M)
    return matrix

# Parameters
M = 200 # Number of rows
K = 3  # Number of states
# Create the true allele frequencies
p = create_random_matrix(M, K)
print(p)
# Parameters
q = np.array([0.4, 0.1, 0.5])  # Initial probabilities for the first K-1 states
r = 0.5  # Parameter influencing transition probabilities
d_values = [0.1]*M  # Example time-dependent values for d_t
# Simulate the time-dependent Markov chain



def simulate_markov_chain_diploid(K, q, r, M, d_values, p):
    x1, q1 = simulate_markov_chain(K, q, r, M, d_values, p)
    x2, q2 = simulate_markov_chain(K, q, r, M, d_values, p)
    return x1, x2

print(simulate_markov_chain_diploid(K, q, r, M, d_values, p))

