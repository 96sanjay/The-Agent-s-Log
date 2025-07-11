
import gymnasium as gym
import numpy as np
import time # To see the agent move more slowly

# Section 1: The Value Iteration logic (unchanged)
def value_iteration(env):
    value_table = np.zeros(16)
    while True:
        max_change = 0
        for square in range(16):
            old_value = value_table[square]
            action_values = []
            for action in range(4):
                total_reward = 0
                for prob, next_square, reward, is_goal in env.unwrapped.P[square][action]:
                    total_reward += prob * (reward + 0.99 * value_table[next_square]) # gamma=0.99
                action_values.append(total_reward)
            value_table[square] = max(action_values)
            max_change = max(max_change, abs(value_table[square] - old_value))
        if max_change < 0.0001:
            break
    return value_table

# Section 2: The Policy Extraction logic (unchanged)
def extract_policy(env, value_table):
    policy = np.zeros(16, dtype=int)
    for square in range(16):
        action_values = []
        for action in range(4):
            total_reward = 0
            for prob, next_square, reward, is_goal in env.unwrapped.P[square][action]:
                total_reward += prob * (reward + 0.99 * value_table[next_square]) # gamma=0.99
            action_values.append(total_reward)
        policy[square] = np.argmax(action_values)
    return policy

# --- Main Execution ---

# Section 3: Solve the MDP to get the policy (mostly unchanged)
# Create the base environment to get its P-model
base_env = gym.make("FrozenLake-v1", is_slippery=True)
print("Solving the MDP...")
optimal_value_table = value_iteration(base_env)
optimal_policy = extract_policy(base_env, optimal_value_table)
print("Optimal Policy Found!")
base_env.close() # We are done with the base env

# --------------------------------------------------------------------
# Section 4: Create Video of the Agent in Action (NEW PART)
# --------------------------------------------------------------------
print("\nRecording video...")

# Create a new environment, but this time enable human rendering
# and wrap it with the video recorder.
video_env = gym.make("FrozenLake-v1", is_slippery=True, render_mode="rgb_array")
video_env = gym.wrappers.RecordVideo(video_env, video_folder="videos")

# Reset the environment to get the starting state
state, info = video_env.reset()

# Loop until the agent falls in a hole or reaches the goal
done = False
while not done:
    # Get the best action for the current state from our policy
    action = optimal_policy[state]

    # Perform the action
    state, reward, terminated, truncated, info = video_env.step(action)
    
    # The episode is done if it's terminated (goal/hole) or truncated (timeout)
    done = terminated or truncated

# Important: Close the environment to save the video file properly.
video_env.close()
print("Video saved in the 'videos' folder!")