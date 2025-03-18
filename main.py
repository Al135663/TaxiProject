import numpy as np
import random
import gymnasium as gym

# Initialize the Taxi-v3 environment
env = gym.make('Taxi-v3')

# Hyperparameters
alpha = 0.1        # Learning rate (how much we update the Q-value)
gamma = 0.95       # Discount factor (importance of future rewards)
epsilon = 1.0      # Initial exploration rate
epsilon_decay = 0.9995  # How much epsilon decreases per episode
min_epsilon = 0.1  # Minimum exploration rate
num_episodes = 10000  # Number of training episodes
max_steps = 100    # Maximum steps per episode

# Initialize the Q-table with zeros (states x actions)
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Function to choose an action based on epsilon-greedy policy
def choose_action(state):
    # With probability epsilon, choose a random action (exploration)
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    # Otherwise, choose the action with the highest Q-value (exploitation)
    else:
        return np.argmax(q_table[state, :])

# Q-learning algorithm
for episode in range(num_episodes):
    # Reset the environment and get the initial state
    state, _ = env.reset()
    done = False

    # Loop through each step in the episode
    for step in range(max_steps):
        # Choose an action using the epsilon-greedy policy
        action = choose_action(state)
        
        # Perform the action and get the next state and reward
        next_state, reward, done, truncated, _ = env.step(action)  
        
        # Get the current Q-value
        old_value = q_table[state, action]
        # Get the maximum Q-value for the next state (best future action)
        next_max = np.max(q_table[next_state, :])
        
        # Q-learning update rule
        q_table[state, action] = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        
        # Move to the next state
        state = next_state
        
        # End the episode if the taxi reaches the destination
        if done or truncated:
            break

    # Decay the exploration rate
    epsilon = max(min_epsilon, epsilon * epsilon_decay) 

# Load the environment again to visualize the agent's performance
env = gym.make('Taxi-v3', render_mode='human')

# Run the trained agent for 5 episodes
for episode in range(5):
    # Reset the environment
    state, _ = env.reset()
    done = False
    print("Episode: ", episode)

    # Step through the environment using the learned policy
    for step in range(max_steps):
        # Render the environment
        env.render()
        
        # Choose the best action from the Q-table
        action = np.argmax(q_table[state, :])
        
        # Take the action and move to the next state
        next_state, reward, done, truncated, _ = env.step(action)
        
        # Update the current state
        state = next_state
        
        # Stop if the episode is finished
        if done or truncated:
            print('Finished episode', episode, 'with reward', reward)
            break

# Close the environment after execution
env.close()
