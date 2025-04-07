import numpy as np
import random
import gymnasium as gym
import time

# Initialize the Taxi-v3 environment with visualization
env = gym.make('Taxi-v3', render_mode='human')

# Hyperparameters
alpha = 0.1
gamma = 0.95
epsilon = 1.0
epsilon_decay = 0.9995
min_epsilon = 0.1
num_episodes = 1000    
max_steps = 100

# Initialize Q-table
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Function to choose action (epsilon-greedy)
def choose_action(state, epsilon):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(q_table[state, :])

# Training loop
for episode in range(num_episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0

    print(f"\nEpisode: {episode+1}")

    for step in range(max_steps):
        # Choose action
        action = choose_action(state, epsilon)

        # Take action
        next_state, reward, done, truncated, _ = env.step(action)

        # Q-learning update
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state, :])
        q_table[state, action] = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)

        state = next_state
        total_reward += reward

        # Slow down so you can see what's happening
        time.sleep(0.05)

        if done or truncated:
            print(f"Finished after {step+1} steps with reward {reward}")
            break

    # Decay epsilon
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

print("\nTraining finished!")
env.close()