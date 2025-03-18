import numpy as np

class EpsilonGreedyAgent:
    def __init__(self, n_arms, epsilon=0.1, initial_value=0):
        self.n_arms = n_arms
        self.epsilon = epsilon
        # Initialize estimated values for each arm
        self.q_values = np.ones(n_arms) * initial_value
        # Count of arm pulls
        self.arm_counts = np.zeros(n_arms)
        
    def select_arm(self):
        # Explore with probability epsilon
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_arms)
        # Exploit with probability 1-epsilon
        else:
            return np.argmax(self.q_values)
    
    def update(self, arm, reward):
        # Update estimated value using incremental average
        self.arm_counts[arm] += 1
        self.q_values[arm] += (reward - self.q_values[arm]) / self.arm_counts[arm]

class SoftmaxAgent:
    def __init__(self, n_arms, temperature=1.0, initial_value=0):
        self.n_arms = n_arms
        self.temperature = temperature  # Controls exploration (higher = more exploration)
        self.q_values = np.ones(n_arms) * initial_value
        self.arm_counts = np.zeros(n_arms)
        
    def select_arm(self):
        # Apply softmax (Boltzmann distribution) to current q-values
        # The temperature parameter controls the amount of exploration
        exp_values = np.exp(self.q_values / self.temperature)
        probabilities = exp_values / np.sum(exp_values)
        
        # Choose arm based on the probabilities
        return np.random.choice(self.n_arms, p=probabilities)
    
    def update(self, arm, reward):
        self.arm_counts[arm] += 1
        self.q_values[arm] += (reward - self.q_values[arm]) / self.arm_counts[arm]

class UCBAgent:
    def __init__(self, n_arms, c=2.0, initial_value=0):
        self.n_arms = n_arms
        self.c = c  # Exploration parameter
        self.q_values = np.ones(n_arms) * initial_value
        self.arm_counts = np.zeros(n_arms)
        self.t = 0  # Total number of plays
        
    def select_arm(self):
        self.t += 1
        
        # For arms that haven't been tried yet, prioritize them
        untried_arms = np.where(self.arm_counts == 0)[0]
        if len(untried_arms) > 0:
            return untried_arms[0]
        
        # UCB formula: Q(a) + c * sqrt(ln(t) / N(a))
        # Where:
        # - Q(a) is the estimated value of arm a
        # - c is the exploration parameter
        # - t is the total number of plays
        # - N(a) is the number of times arm a has been played
        
        ucb_values = self.q_values + self.c * np.sqrt(
            np.log(self.t) / self.arm_counts
        )
        return np.argmax(ucb_values)
    
    def update(self, arm, reward):
        self.arm_counts[arm] += 1
        self.q_values[arm] += (reward - self.q_values[arm]) / self.arm_counts[arm]

class ThompsonSamplingAgent:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        # For each arm, we maintain a Beta distribution with parameters alpha and beta
        # Alpha represents successful pulls (rewards), beta represents unsuccessful pulls
        self.alphas = np.ones(n_arms)  # Prior: 1 success
        self.betas = np.ones(n_arms)   # Prior: 1 failure
        
    def select_arm(self):
        # Sample from each arm's Beta distribution
        samples = np.random.beta(self.alphas, self.betas)
        # Choose the arm with the highest sample
        return np.argmax(samples)
    
    def update(self, arm, reward):
        # For the Gaussian reward case, we need to normalize between 0 and 1
        # For simplicity, we'll treat rewards > 0 as success
        if reward > 0:
            self.alphas[arm] += 1
        else:
            self.betas[arm] += 1