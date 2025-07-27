
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from math import factorial
import time

# --- 1. Environment and Problem Setup ---

# Maximum number of cars at each location
MAX_CARS = 20
# Maximum number of cars Jack can move
MAX_MOVE = 5
# Reward for renting a car
RENTAL_REWARD = 10
# Cost for moving a car
MOVE_COST = 2

# Average number of requests and returns (the "pattern")
AVG_REQUESTS_LOC1 = 3
AVG_REQUESTS_LOC2 = 4
AVG_RETURNS_LOC1 = 3
AVG_RETURNS_LOC2 = 2

# Discount factor
GAMMA = 0.9

# Create the state space: all combinations of cars at both locations
states = []
for i in range(MAX_CARS + 1):
    for j in range(MAX_CARS + 1):
        states.append((i, j))

# Create the action space: from -5 to +5
# An action of +2 means move 2 cars from loc 1 to 2
# An action of -3 means move 3 cars from loc 2 to 1
actions = np.arange(-MAX_MOVE, MAX_MOVE + 1)

# --- 2. Poisson Probability Function ---

# A cache to store computed Poisson probabilities to speed things up
poisson_cache = dict()

def poisson(n, lam):
    """
    Calculates the probability of n occurrences given an average rate (lambda).
    Uses a cache to avoid re-calculating the same values.
    """
    global poisson_cache
    if (n, lam) in poisson_cache:
        return poisson_cache[(n, lam)]
    
    # Formula for Poisson Probability Distribution
    p = (lam**n * np.exp(-lam)) / factorial(n)
    poisson_cache[(n, lam)] = p
    return p

# --- 3. Helper function for a state-action value ---

def calculate_expected_value(state, action, value_table):
    """
    Calculates the expected value of a state-action pair.
    This is the core of the Bellman equation application.
    """
    cars1, cars2 = state
    
    # Cost for moving cars
    expected_reward = -MOVE_COST * abs(action)
    
    # Number of cars after moving them overnight
    cars1_after_move = int(min(cars1 - action, MAX_CARS))
    cars2_after_move = int(min(cars2 + action, MAX_CARS))
    
    # Iterate through all possible numbers of rental requests
    # We cap at 11, as the probability of more is negligible
    for req1 in range(11):
        for req2 in range(11):
            # Probability of this combination of requests
            prob_req = poisson(req1, AVG_REQUESTS_LOC1) * poisson(req2, AVG_REQUESTS_LOC2)
            
            # Number of cars actually rented
            rentals1 = min(cars1_after_move, req1)
            rentals2 = min(cars2_after_move, req2)
            
            # Calculate the immediate reward from rentals
            reward = (rentals1 + rentals2) * RENTAL_REWARD
            
            # Number of cars left at the end of the day
            cars_left1 = cars1_after_move - rentals1
            cars_left2 = cars2_after_move - rentals2
            
            # Now, iterate through all possible numbers of returns
            for ret1 in range(11):
                for ret2 in range(11):
                    # Probability of this combination of returns
                    prob_ret = poisson(ret1, AVG_RETURNS_LOC1) * poisson(ret2, AVG_RETURNS_LOC2)
                    
                    # Total probability for this scenario (requests and returns)
                    prob = prob_req * prob_ret
                    
                    # Final number of cars for the next day's state
                    final_cars1 = min(cars_left1 + ret1, MAX_CARS)
                    final_cars2 = min(cars_left2 + ret2, MAX_CARS)
                    
                    # Add the discounted value of the next state to the total expected reward
                    expected_reward += prob * (reward + GAMMA * value_table[final_cars1, final_cars2])

    return expected_reward

# --- 4. Main Policy Iteration Algorithm ---

def policy_iteration():
    # Initialize a policy of doing nothing (action=0)
    policy = np.zeros((MAX_CARS + 1, MAX_CARS + 1), dtype=int)
    
    # Initialize the value table to all zeros
    value_table = np.zeros((MAX_CARS + 1, MAX_CARS + 1))

    iteration = 1
    # --- Main Loop ---
    while True:
        print(f"--- Starting Iteration {iteration} ---")
        
        # --- a. Policy Evaluation ---
        # Run until the value table stabilizes
        eval_iter = 1
        while True:
            max_change = 0
            for cars1, cars2 in states:
                old_value = value_table[cars1, cars2]
                
                # Get the action from the current policy
                action = policy[cars1, cars2]
                
                # Calculate the new value for this state under the current policy
                new_value = calculate_expected_value((cars1, cars2), action, value_table)
                
                value_table[cars1, cars2] = new_value
                max_change = max(max_change, abs(old_value - new_value))
            
            print(f"  Policy Eval Iteration {eval_iter}, Max Change: {max_change:.2f}")
            eval_iter += 1

            # Check for convergence
            if max_change < 1e-2: # A less strict threshold for speed
                print("  Policy evaluation converged.")
                break

        # --- b. Policy Improvement ---
        policy_stable = True
        print("Improving policy...")
        for cars1, cars2 in states:
            old_action = policy[cars1, cars2]
            
            action_values = []
            # Check all possible actions
            for action in actions:
                # Ensure the action is valid (can't move more cars than available)
                if (action >= 0 and cars1 >= action) or (action < 0 and cars2 >= -action):
                    action_values.append(calculate_expected_value((cars1, cars2), action, value_table))
                else:
                    # Assign a very low value to invalid actions
                    action_values.append(-np.inf)
            
            # Find the best action
            best_action_idx = np.argmax(action_values)
            best_action = actions[best_action_idx]
            
            policy[cars1, cars2] = best_action
            
            # Check if the policy has changed
            if old_action != best_action:
                policy_stable = False
        
        # If the policy is stable, we have found the optimal policy
        if policy_stable:
            print("\nOptimal policy found!")
            break
        
        iteration += 1
            
    return policy, value_table


# --- Run the algorithm ---
start_time = time.time()
final_policy, final_value = policy_iteration()
print(f"\nAlgorithm finished in {time.time() - start_time:.2f} seconds.")

# --- Visualize the final policy ---
fig, ax = plt.subplots(figsize=(12, 10))
ax = sns.heatmap(final_policy, annot=False, ax=ax)
ax.invert_yaxis()
ax.set_title("Optimal Policy for Jack's Car Rental (Action: Cars moved from Loc 1 to Loc 2)")
ax.set_xlabel("Cars at Location 2")
ax.set_ylabel("Cars at Location 1")
plt.show()

# --- Visualize the final value function ---
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(np.arange(MAX_CARS + 1), np.arange(MAX_CARS + 1))
ax.plot_surface(X, Y, final_value, cmap='viridis')
ax.set_xlabel("Cars at Location 2")
ax.set_ylabel("Cars at Location 1")
ax.set_zlabel("Value")
ax.set_title("Optimal Value Function")
plt.show()