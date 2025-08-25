import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from env import RubiksCube


# Neural network model for approximating Q-values
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# Create the Cube environment
env = RubiksCube()

n = 2

epsilon_decay_array = [0.995, 0.9995]
batch_size_array = [96, 192]
target_update_freq_array = [1500, 3000]
memory_size_array = [15000, 30000]
episodes_array = [1500, 3000]


# Initialize Q-networks
learning_rate = 0.001
input_dim = len(env.flatten())
output_dim = env.nA
policy_net = DQN(input_dim, output_dim)
target_net = DQN(input_dim, output_dim)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)

for iteration in range(n):
    i = random.randint(0, iteration)
    env.set_nMix(i+1)

    # Hyperparameters
    # learning_rate = 0.001
    # gamma = 0.99
    # epsilon = 1.0
    # epsilon_min = 0.01
    # epsilon_decay = 0.995
    # batch_size = 64
    # target_update_freq = 1000
    # memory_size = 10000
    # episodes = 1000

    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.001
    epsilon_decay = epsilon_decay_array[i]
    batch_size = batch_size_array[i]
    target_update_freq = target_update_freq_array[i]
    memory_size = memory_size_array[i]
    episodes = episodes_array[i]

    memory = deque(maxlen=memory_size)

    # Function to choose action using epsilon-greedy policy
    def select_action(state, epsilon):
        if random.random() < epsilon:
            # choose a random action
            return env.sample_action()
        else:
            # Choose the action with the highest Q-value
            state = torch.FloatTensor(state).unsqueeze(0)
            q_values = policy_net(state)
            return torch.argmax(q_values).item()  # Exploit

    # Function to optimize the model using experience replay
    def optimize_model():
        if len(memory) < batch_size:
            return
        
        batch = random.sample(memory, batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

        state_batch = torch.FloatTensor(np.array(state_batch))
        action_batch = torch.LongTensor(np.array(action_batch)).unsqueeze(1)
        reward_batch = torch.FloatTensor(np.array(reward_batch))
        next_state_batch = torch.FloatTensor(np.array(next_state_batch))
        done_batch = torch.FloatTensor(np.array(done_batch))

        # Compute Q-values for current states
        q_values = policy_net(state_batch).gather(1, action_batch).squeeze()

        # Compute target Q-values using the target network
        with torch.no_grad():
            max_next_q_values = target_net(next_state_batch).max(1)[0]
            target_q_values = reward_batch + gamma * max_next_q_values * (1 - done_batch)

        loss = nn.MSELoss()(q_values, target_q_values)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Main training loop
    rewards_per_episode = []
    epsilon_values = []  # Track epsilon values over episodes
    steps_done = 0

    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Select action
            action = select_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            
            # Store transition in memory
            memory.append((state, action, reward, next_state, done))
            
            # Update state
            state = next_state
            episode_reward += reward
            
            # Optimize model
            optimize_model()

            # Update target network periodically
            if steps_done % target_update_freq == 0:
                target_net.load_state_dict(policy_net.state_dict())

            steps_done += 1

        # Decay epsilon
        epsilon = max(epsilon_min, epsilon_decay * epsilon)
        
        rewards_per_episode.append(episode_reward)
        epsilon_values.append(epsilon)  # Store current epsilon value

    # Plotting the rewards and epsilon per episode
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot rewards
    ax1.plot(rewards_per_episode)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('DQN Training - Rewards per Episode')
    ax1.grid(True)

    # Plot epsilon
    ax2.plot(epsilon_values)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Epsilon')
    ax2.set_title('DQN Training - Epsilon Decay')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()