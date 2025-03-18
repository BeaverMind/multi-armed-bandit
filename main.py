import numpy as np
import matplotlib.pyplot as plt
import time

from agents import EpsilonGreedyAgent, SoftmaxAgent, UCBAgent, ThompsonSamplingAgent

class MultiArmedBandit:
    def __init__(self, k_arms=10, true_reward_mean=0, reward_variance=1):
        # Initialize k arms with true reward values from a normal distribution
        self.k = k_arms
        self.true_reward_means = np.random.normal(true_reward_mean, 1, k_arms)
        self.reward_variance = reward_variance
        
    def pull(self, arm):
        # Generate reward when a specific arm is pulled
        return np.random.normal(self.true_reward_means[arm], self.reward_variance)
    
def run_simulation(n_steps=1000, n_bandits=1000):
    rewards = {
        'Epsilon-Greedy': np.zeros(n_steps),
        'Softmax': np.zeros(n_steps),
        'UCB': np.zeros(n_steps),
        'Thompson Sampling': np.zeros(n_steps)
    }
    
    optimal_actions = {
        'Epsilon-Greedy': np.zeros(n_steps),
        'Softmax': np.zeros(n_steps),
        'UCB': np.zeros(n_steps),
        'Thompson Sampling': np.zeros(n_steps)
    }
    
    # Track runtime for each policy
    runtime = {
        'Epsilon-Greedy': 0,
        'Softmax': 0,
        'UCB': 0,
        'Thompson Sampling': 0
    }
    
    for i in range(n_bandits):
        # Create bandit problem
        bandit = MultiArmedBandit()
        optimal_arm = np.argmax(bandit.true_reward_means)
        
        # Create agents with different strategies
        agents = {
            'Epsilon-Greedy': EpsilonGreedyAgent(bandit.k, epsilon=0.1),
            'Softmax': SoftmaxAgent(bandit.k, temperature=0.1),
            'UCB': UCBAgent(bandit.k, c=2.0),
            'Thompson Sampling': ThompsonSamplingAgent(bandit.k)
        }
        
        for t in range(n_steps):
            for name, agent in agents.items():
                # Measure time for this policy's decision and update
                start_time = time.time()
                
                # Select arm and get reward
                arm = agent.select_arm()
                reward = bandit.pull(arm)
                
                # Update agent
                agent.update(arm, reward)
                
                # Record runtime
                runtime[name] += time.time() - start_time
                
                # Record results
                rewards[name][t] += reward
                if arm == optimal_arm:
                    optimal_actions[name][t] += 1
    
    # Average over all bandits
    for name in agents.keys():
        rewards[name] /= n_bandits
        optimal_actions[name] /= n_bandits
    
    return rewards, optimal_actions, runtime

def plot_results(rewards, optimal_actions, runtime):
    plt.figure(figsize=(16, 10))
    
    plt.subplot(2, 2, 1)
    for name, reward in rewards.items():
        plt.plot(reward, label=name)
    plt.xlabel('Steps')
    plt.ylabel('Average reward')
    plt.title('Average Reward over Time')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    for name, opt_action in optimal_actions.items():
        plt.plot(opt_action, label=name)
    plt.xlabel('Steps')
    plt.ylabel('% Optimal action')
    plt.title('Percentage of Optimal Actions over Time')
    plt.legend()
    
    if runtime:
        plt.subplot(2, 2, 3)
        plt.bar(runtime.keys(), runtime.values())
        plt.ylabel('Total runtime (seconds)')
        plt.title('Runtime Comparison')
        plt.xticks(rotation=45)
        
        # Calculate relative performance
        fastest = min(runtime.values())
        relative_speed = {name: time/fastest for name, time in runtime.items()}
        
        plt.subplot(2, 2, 4)
        plt.bar(relative_speed.keys(), relative_speed.values())
        plt.ylabel('Relative runtime (fastest = 1.0)')
        plt.title('Relative Runtime Performance')
        plt.xticks(rotation=45)
        
        for i, (name, value) in enumerate(relative_speed.items()):
            plt.text(i, value + 0.05, f"{value:.2f}x", ha='center')
    
    
    plt.savefig(f'results.pdf', bbox_inches='tight')
    plt.tight_layout()
    plt.show(block=True)

# Run and plot
rewards, optimal_actions, runtime = run_simulation()
plot_results(rewards, optimal_actions, runtime)