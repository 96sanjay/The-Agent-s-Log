---
layout: post
title:  "Solving Jack's Car Rental: A Deep Dive into Dynamic Programming"
date:   2025-07-25 03:55:00 +0200
categories: reinforcement-learning
---
Solving Jack's Car Rental: A Complete Guide to a Solution using Dynamic Programming
1. Introduction: The Problem
The "Jack's Car Rental" problem is a classic challenge in reinforcement learning that models a real-world resource management scenario. A business owner, Jack, runs a car rental service with two locations. The challenge is that customer demand (both for renting and returning cars) is random and differs between the two locations. Every night, Jack has the option to move cars from one location to the other to better prepare for the next day, but moving cars costs money.

The core dilemma is balancing the cost of moving cars against the potential lost revenue from not having a car available for a customer. Our goal is to find a perfect, optimal strategy that tells Jack exactly how many cars to move each night, based on the number of cars at each location, to maximize his long-term profit.

2. The Mathematical Framework: Markov Decision Process (MDP)
To solve this problem rigorously, we first frame it as a Markov Decision Process (MDP). This provides the mathematical foundation to describe the environment and the decisions to be made. An MDP is defined by five key components:

S (States): A state s is a complete snapshot of the environment. Here, a state is the number of cars at Location 1 and Location 2 at the end of the day. With a maximum of 20 cars each, our state space is a 21x21 grid, from (0,0) to (20,20).

A (Actions): An action a is a decision Jack can make. He can move up to 5 cars between locations. We represent this as an integer from -5 (move 5 cars from Loc 2 to 1) to +5 (move 5 cars from Loc 1 to 2).

P (Transition Probabilities): The function P(s 
′
 ∣s,a) gives the probability of moving to a new state s' if we are in state s and take action a. This is where we model the environment's randomness. The number of cars requested and returned follows a Poisson distribution, a tool for modeling the probability of a given number of events happening over a fixed interval when we know the average rate of occurrence. The formula is:
P(n)= 
n!
λ 
n
 ⋅e 
−λ
 
​
 
where λ is the known average rate (e.g., average requests at Location 1 is 3). To find the total probability of a transition, our algorithm must calculate the joint probability of every possible combination of requests and returns at both locations.

R (Rewards): The reward R is the immediate feedback from the environment. Jack receives a +$10 reward for each car rented and a -$2 cost for each car moved. The algorithm's goal is to maximize the total cumulative reward.

γ (Discount Factor): The discount factor γ (we used 0.9) determines the value of future rewards. A reward one day from now is worth 90% of a reward today. This ensures that the long-term profit is a finite, calculable number.

3. The Algorithm: Policy Iteration
To find the optimal strategy, we use Policy Iteration, a powerful algorithm from Dynamic Programming. This algorithm is ideal when we have a full model of the MDP, as we do here. It works by repeating two steps until the policy can no longer be improved.

Step 3a: Policy Evaluation (The "Appraisal")
Goal: To take our current strategy (policy π) and accurately calculate its long-term value, V 
π
​
 (s), for every state s.

The Math: This is done by repeatedly applying the Bellman Expectation Equation. The value of a state under a policy is the expected immediate reward plus the discounted value of all possible next states. We iterate this calculation until the values converge:

V 
k+1
​
 (s)= 
s 
′
 
∑
​
 P(s 
′
 ∣s,π(s))[R(s,π(s),s 
′
 )+γV 
k
​
 (s 
′
 )]
Step 3b: Policy Improvement (The "Strategizing")
Goal: With an accurate value function V 
π
​
  for our current policy, we now create a new, better policy, π 
′
 .

The Math: We act "greedily" with respect to the value function. For each state, we look at every possible action and choose the one that leads to the greatest expected reward. The new policy becomes:

π 
′
 (s)= 
a∈A
argmax
​
  
s 
′
 
∑
​
 P(s 
′
 ∣s,a)[R(s,a,s 
′
 )+γV 
π
​
 (s 
′
 )]
The algorithm repeats these two steps. The new, improved policy from step 3b becomes the policy to be evaluated in the next cycle. This continues until the policy no longer changes, at which point it is optimal. This convergence is mathematically guaranteed by the Contraction Mapping Theorem.

4. The Code Implementation: A Walkthrough
Setup and Helpers
The first part of the code sets up the constants (MAX_CARS, RENTAL_REWARD, etc.) and a poisson() helper function to calculate probabilities on demand, using a cache to avoid recomputing and save time.

The Core Engine: calculate_expected_value function
This function is the implementation of the right-hand side of the Bellman equation. Its job is to calculate the total expected value (the score) if you are in a certain state and take a certain action. It does this by:

Calculating the immediate cost of moving cars.

Looping through every plausible number of rental requests for both locations.

Looping through every plausible number of car returns for both locations.

Inside these nested loops, for each specific scenario, it calculates:

The total probability of this exact combination of random events happening.

The rental income from this scenario.

The score for this one scenario, which is probability * (income + discounted_value_of_next_state).

It sums the scores from all these thousands of tiny scenarios to get the final, total expected value.

The Main Algorithm: policy_iteration function
This function manages the entire learning process.

Initialization: It creates a starting policy (e.g., "always move 0 cars") and a value table filled with zeros.

Main Loop: It enters a while True: loop that alternates between the two phases:

Policy Evaluation: This phase has its own inner while loop. It repeatedly calls calculate_expected_value for every state (using the current policy's action) until the value_table stabilizes.

Policy Improvement: Once the values are accurate, this phase loops through every state and calls calculate_expected_value for every possible action (-5 to +5) to find the best one. It updates the policy with this new best action.

Termination: After the improvement phase, it checks if the policy has changed. If it has, the main loop repeats. If not, the policy is optimal, and the loop breaks.

5. The Results: Interpreting the Solution
The algorithm produces two key outputs:

The Optimal Policy Heatmap: This is the final "cheat sheet" for Jack. The x- and y-axes represent the number of cars at each location, and the color of each square shows the optimal number of cars to move. The plot clearly shows a strategy of moving cars from a location with a surplus to one with a deficit to achieve a profitable balance.

The Optimal Value Function 3D Plot: This is the final "scorecard." The height of the surface at any point shows the maximum expected long-term profit Jack can make starting with that combination of cars. The plot slopes upwards, confirming that having more cars in the system is more valuable. The policy's goal is to always take actions that move Jack "uphill" on this value surface.