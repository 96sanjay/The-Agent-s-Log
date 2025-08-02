
import gymnasium as gym
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Setup ---

# Simulation parameters
EPISODES = 20000
LEARNING_RATE = 0.1
GAMMA = 0.99
# How often to log the error
LOG_INTERVAL = 1000

# Create the Frozen Lake environment
env = gym.make("FrozenLake-v1", is_slippery=True)
n_states = env.observation_space.n
n_actions = env.action_space.n

# Define a simple, fixed policy to evaluate
def simple_policy(state):
    return 2 # Always "RIGHT"

# --- 2. Ground Truth Calculation (using Dynamic Programming) ---

def policy_evaluation_dp(policy_func):
    """Calculates the true value function for a policy using DP."""
    value_table = np.zeros(n_states)
    theta = 1e-8
    while True:
        delta = 0
        for s in range(n_states):
            v_old = value_table[s]
            a = policy_func(s)
            # Calculate expected value for the action
            new_val = 0
            for prob, next_state, reward, done in env.unwrapped.P[s][a]:
                new_val += prob * (reward + GAMMA * value_table[next_state])
            value_table[s] = new_val
            delta = max(delta, abs(v_old - value_table[s]))
        if delta < theta:
            break
    return value_table

# --- 3. Model-Free Prediction Functions (Modified to track error) ---

def monte_carlo_prediction(true_value_table):
    """Estimates V using Monte Carlo and tracks RMSE over episodes."""
    value_table = np.zeros(n_states)
    rmse_history = []
    
    print("--- Running Monte Carlo (MC) Prediction ---")
    for episode in range(EPISODES):
        episode_history = []
        state, info = env.reset()
        done = False
        while not done:
            action = simple_policy(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_history.append((state, reward))
            state = next_state
        
        G = 0
        for state, reward in reversed(episode_history):
            G = reward + GAMMA * G
            value_table[state] += LEARNING_RATE * (G - value_table[state])
            
        # Log RMSE at intervals
        if episode % LOG_INTERVAL == 0:
            rmse = np.sqrt(np.mean((value_table - true_value_table)**2))
            rmse_history.append(rmse)
            
    return value_table, rmse_history

def td_prediction(true_value_table):
    """Estimates V using TD(0) and tracks RMSE over episodes."""
    value_table = np.zeros(n_states)
    rmse_history = []

    print("--- Running Temporal-Difference (TD) Prediction ---")
    for episode in range(EPISODES):
        state, info = env.reset()
        done = False
        while not done:
            action = simple_policy(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            td_target = reward + GAMMA * value_table[next_state]
            value_table[state] += LEARNING_RATE * (td_target - value_table[state])
            state = next_state
        
        # Log RMSE at intervals
        if episode % LOG_INTERVAL == 0:
            rmse = np.sqrt(np.mean((value_table - true_value_table)**2))
            rmse_history.append(rmse)
            
    return value_table, rmse_history

# --- 4. Policy Extraction and Visualization Functions ---

def extract_policy(value_table):
    """Creates a greedy policy from a value table."""
    policy = np.zeros(n_states, dtype=int)
    for state in range(n_states):
        q_values = [sum([p * (r + GAMMA * value_table[s_]) for p, s_, r, _ in env.unwrapped.P[state][action]]) for action in range(n_actions)]
        policy[state] = np.argmax(q_values)
    return policy

def plot_learning_curve(mc_history, td_history):
    """Plots the Root Mean Squared Error over episodes."""
    plt.figure(figsize=(12, 6))
    episodes = np.arange(len(mc_history)) * LOG_INTERVAL
    plt.plot(episodes, mc_history, label='Monte Carlo')
    plt.plot(episodes, td_history, label='Temporal-Difference (TD)')
    plt.title('Learning Curve: Value Function Error vs. Episodes')
    plt.xlabel('Episodes')
    plt.ylabel('Root Mean Squared Error (RMSE)')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_value_heatmap(title, value_table):
    """Visualizes a value table as a heatmap."""
    plt.figure(figsize=(8, 8))
    sns.heatmap(value_table.reshape(4, 4), annot=True, fmt=".3f", cmap="viridis", cbar=False)
    plt.title(title)
    plt.show()

def plot_policy_arrows(title, policy):
    """Visualizes a policy with arrows on the grid."""
    grid_size = 4
    policy_grid = policy.reshape(grid_size, grid_size)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.matshow(np.zeros((grid_size, grid_size)), cmap='binary')
    
    arrows = ['←', '↓', '→', '↑']
    for i in range(grid_size):
        for j in range(grid_size):
            ax.text(j, i, arrows[policy_grid[i, j]], ha='center', va='center', fontsize=20)
    
    ax.set_xticks(np.arange(grid_size))
    ax.set_yticks(np.arange(grid_size))
    ax.set_xticklabels(np.arange(grid_size))
    ax.set_yticklabels(np.arange(grid_size))
    ax.set_title(title)
    plt.show()

# --- 5. Main Execution and Comparison ---

if __name__ == "__main__":
    # Calculate the true value function to compare against
    true_values = policy_evaluation_dp(simple_policy)
    
    # Run MC and TD and get their learning histories
    mc_values, mc_rmse = monte_carlo_prediction(true_values)
    td_values, td_rmse = td_prediction(true_values)
    
    # Extract the final policy from the (usually better) TD values
    optimal_policy = extract_policy(td_values)
    
    # --- Show the Visualizations ---
    plot_learning_curve(mc_rmse, td_rmse)
    plot_value_heatmap("TD Final Value Function", td_values)
    plot_policy_arrows("Final Extracted Policy", optimal_policy)

    # --- Record a Video ---
    # ... (video recording code remains the same) ...

    env.close()