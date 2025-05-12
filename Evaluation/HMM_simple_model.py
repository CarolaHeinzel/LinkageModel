import numpy as np

# 1) Simulating HMM data
def simulate_hmm(theta, n_steps=100):
    n_states = 2  # Number of hidden states
    n_obs = 2     # Number of possible observations

    # Transition probabilities matrix (state-to-state transitions)
    transition_probs = np.array([[0.8 - 0.4 * theta, 0.2 + 0.4 * theta],
                                 [0.3 + 0.4 * theta, 0.7 - 0.4 * theta]])

    # Emission probabilities matrix (state-to-observation emissions)
    emission_probs = np.array([[0.9 - 0.4 * theta, 0.1 + 0.4 * theta],
                               [0.2 + 0.4 * theta, 0.8 - 0.4 * theta]])

    start_probs = np.array([0.6, 0.4])  # Initial state probabilities

    # Initialize arrays for states and observations
    states = np.zeros(n_steps, dtype=int)
    observations = np.zeros(n_steps, dtype=int)

    # Assign the initial state based on start probabilities
    states[0] = np.random.choice(n_states, p=start_probs)

    # Simulate the hidden states and corresponding observations
    for t in range(1, n_steps):
        states[t] = np.random.choice(n_states, p=transition_probs[states[t-1]])
        observations[t] = np.random.choice(n_obs, p=emission_probs[states[t]])

    return states, observations

# 2) Baum-Welch Algorithm (Expectation-Maximization)
def forward_algorithm(transition_probs, emission_probs, start_probs, observations, n_steps):
    n_states = len(start_probs)
    n_obs = len(emission_probs[0])

    alpha = np.zeros((n_steps, n_states))

    # Calculate the initial alpha(0) values based on the start probabilities
    for i in range(n_states):
        alpha[0, i] = start_probs[i] * emission_probs[i, observations[0]]

    # Calculate the remaining alpha(t) values (forward recursion)
    for t in range(1, n_steps):
        for i in range(n_states):
            alpha[t, i] = np.sum(alpha[t-1] * transition_probs[:, i]) * emission_probs[i, observations[t]]

    return alpha

def backward_algorithm(transition_probs, emission_probs, observations, n_steps):
    n_states = len(transition_probs)
    beta = np.zeros((n_steps, n_states))

    # Set the final beta(T-1) values to 1
    beta[n_steps-1] = 1

    # Calculate the remaining beta(t) values (backward recursion)
    for t in range(n_steps-2, -1, -1):
        for i in range(n_states):
            beta[t, i] = np.sum(transition_probs[i, :] * emission_probs[:, observations[t+1]] * beta[t+1])

    return beta

def baum_welch(observations, n_steps, transition_probs, emission_probs, start_probs, tol=1e-10):
    n_states = 2
    n_obs = 2

    # Iterative optimization (Expectation-Maximization)
    iteration = 0
    while True:
        iteration += 1
        
        # Step 1: Calculate the forward and backward probabilities
        alpha = forward_algorithm(transition_probs, emission_probs, start_probs, observations, n_steps)
        beta = backward_algorithm(transition_probs, emission_probs, observations, n_steps)

        # Step 2: Calculate the gamma and xi values (expectations)
        gamma = np.zeros((n_steps, n_states))
        xi = np.zeros((n_steps - 1, n_states, n_states))

        for t in range(n_steps):
            denom = np.sum(alpha[t] * beta[t])
            for i in range(n_states):
                gamma[t, i] = (alpha[t, i] * beta[t, i]) / denom

        for t in range(n_steps - 1):
            denom = np.sum(np.dot(alpha[t, :], transition_probs) * beta[t + 1, :] * emission_probs[:, observations[t + 1]])
            for i in range(n_states):
                for j in range(n_states):
                    xi[t, i, j] = (alpha[t, i] * transition_probs[i, j] * emission_probs[j, observations[t + 1]] * beta[t + 1, j]) / denom

        # Step 3: Maximization of transition and emission probabilities
        # Maximizing transition probabilities
        transition_probs_new = np.sum(xi, axis=0) / np.sum(gamma[:-1], axis=0)[:, None]

        # Maximizing emission probabilities
        emission_probs_new = np.zeros_like(emission_probs)
        for i in range(n_states):
            for t in range(n_steps):
                if observations[t] == 0:
                    emission_probs_new[i, 0] += gamma[t, i]
                else:
                    emission_probs_new[i, 1] += gamma[t, i]
        emission_probs_new = emission_probs_new / np.sum(gamma, axis=0)[:, None]

        # Check for convergence: If the change in transition and emission probabilities is minimal, stop
        if np.max(np.abs(transition_probs_new - transition_probs)) < tol and np.max(np.abs(emission_probs_new - emission_probs)) < tol:
            print(f"Converged at iteration {iteration}")
            break

        # Update parameters for the next iteration
        transition_probs = transition_probs_new
        emission_probs = emission_probs_new

    return transition_probs, emission_probs

# 4) Simulate HMM Data with a true parameter theta
theta = 0.1
n_steps = 1000
true_states, simulated_observations = simulate_hmm(theta, n_steps)

# 5) Estimate transition and emission probabilities using Baum-Welch
transition_probs_est, emission_probs_est = baum_welch(simulated_observations, n_steps, 
                                                       np.array([[0.8, 0.2],
                                                                 [0.3, 0.7]]), 
                                                       np.array([[0.5, 0.5], [0.5, 0.5]]), 
                                                       np.array([0.9, 0.1]))

# Define the true transition and emission probabilities for comparison
transition_probs_true = np.array([[0.8 + theta, 0.2 - theta],
                                  [0.3, 0.7]])

emission_probs_true = np.array([[0.9 - 0.4 * theta, 0.1 + 0.4 * theta],
                                [0.2 + 0.4 * theta, 0.8 - 0.4 * theta]])

# Print the results
print("True transition probabilities:")
print(transition_probs_true)
print("Estimated transition probabilities:")
print(transition_probs_est)

print("\nTrue emission probabilities:")
print(emission_probs_true)
print("Estimated emission probabilities:")
print(emission_probs_est)
