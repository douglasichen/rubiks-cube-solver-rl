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
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# Create the Cube environment
env = RubiksCube()

n = 3

k=9
k2=k*k
epsilon_decay_array = [0.995, 0.9996, 0.99996]

# epsilon_decay_array = [0.995, 1]
batch_size_array = [96, 192*k, 192*k2]
target_update_freq_array = [1500, 2000*k, 2000*k2]
memory_size_array = [15000, 20000*k, 20000*k2]
episodes_array = [1500, 2000*k, 2000*k2]


# Initialize Q-networks
learning_rate = 0.001
input_dim = len(env.flatten())
output_dim = env.nA
policy_net = DQN(input_dim, output_dim)
target_net = DQN(input_dim, output_dim)

# move to mps to use gpu
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
policy_net = policy_net.to(device)
target_net = target_net.to(device)
print(f"Policy net device: {next(policy_net.parameters()).device}")


target_net.load_state_dict(policy_net.state_dict())
target_net.eval()



optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)

def get_random_mix_number(n: int):
    weights = list(range(1, n + 2))
    numbers = list(range(n + 1))
    return random.choices(numbers, weights=weights)[0]

for iteration in range(n):
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
    epsilon_min = 0.00001

    memory = deque(maxlen=memory_size_array[iteration])
    # memory = deque(memory, maxlen=memory_size_array[iteration])

    # Function to choose action using epsilon-greedy policy
    def select_action(state, epsilon):
        if random.random() < epsilon:
            # choose a random action
            return env.sample_action()
        else:
            # Choose the action with the highest Q-value
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = policy_net(state)
            return torch.argmax(q_values).item()  # Exploit

    # Function to optimize the model using experience replay
    def optimize_model():
        if len(memory) < batch_size_array[iteration]:
            return
        
        batch = random.sample(memory, batch_size_array[iteration])
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

        state_batch = torch.FloatTensor(np.array(state_batch)).to(device)
        action_batch = torch.LongTensor(np.array(action_batch)).unsqueeze(1).to(device)
        reward_batch = torch.FloatTensor(np.array(reward_batch)).to(device)
        next_state_batch = torch.FloatTensor(np.array(next_state_batch)).to(device)
        done_batch = torch.FloatTensor(np.array(done_batch)).to(device)

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

    # Track rewards and epsilon for each mix number separately
    mix_rewards = {}  # Dictionary to store rewards for each mix number
    mix_epsilon = {}  # Dictionary to store epsilon for each mix number

    for episode in range(episodes_array[iteration]):
        # must randomly distribute i so the model does not forget how to solve the simpler mixes. but use a smaller i less often
        random_mix_number = get_random_mix_number(iteration) + 1
        env.set_nMix(random_mix_number)

        state, mixed_actions = env.reset()
        episode_reward = 0
        done = False
        
        actions = []

        while not done:
            # Select action
            action = select_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            actions.append(action)
            
            # Store transition in memory
            memory.append((state, action, reward, next_state, done))
            
            # Update state
            state = next_state
            episode_reward += reward
            
            # Optimize model
            optimize_model()

            # Update target network periodically
            if steps_done % target_update_freq_array[iteration] == 0:
                target_net.load_state_dict(policy_net.state_dict())

            steps_done += 1

        # if random_mix_number == 2:
        #     print(mixed_actions)
        #     print(actions)
        #     if mixed_actions[0] == actions[1]+1 and mixed_actions[1] == actions[0]+1:
        #         print("THIS SHULD BE WORKING", mixed_actions, actions)

        # Decay epsilon
        epsilon = max(epsilon_min, epsilon_decay_array[iteration] * epsilon)
        
        rewards_per_episode.append(episode_reward)
        epsilon_values.append(epsilon)  # Store current epsilon value
        
        # Store data for current mix number
        if random_mix_number not in mix_rewards:
            mix_rewards[random_mix_number] = []
            mix_epsilon[random_mix_number] = []
        mix_rewards[random_mix_number].append(episode_reward)
        mix_epsilon[random_mix_number].append(epsilon)

    if iteration != 2:
        continue
    # Plotting the rewards and epsilon per episode
    import matplotlib.pyplot as plt

    # Plot separate graphs for each mix number
    num_mixes = len(mix_rewards)
    fig2, axes = plt.subplots(num_mixes, 2, figsize=(12, 4*num_mixes))
    
    if num_mixes == 1:
        axes = axes.reshape(1, -1)
    
    for j, (mix_num, rewards) in enumerate(mix_rewards.items()):
        # Plot rewards for this mix number
        axes[j, 0].plot(rewards)
        axes[j, 0].set_xlabel('Episode')
        axes[j, 0].set_ylabel('Reward')
        axes[j, 0].set_title(f'Mix {mix_num} - Rewards per Episode')
        axes[j, 0].grid(True)
        
        # Plot epsilon for this mix number
        axes[j, 1].plot(mix_epsilon[mix_num])
        axes[j, 1].set_xlabel('Episode')
        axes[j, 1].set_ylabel('Epsilon')
        axes[j, 1].set_title(f'Mix {mix_num} - Epsilon Decay')
        axes[j, 1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
torch.save(policy_net, 'model.pth')