import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from env import RubiksCube
import matplotlib.pyplot as plt
from tqdm import tqdm


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


def calc_epsilon(episode_start: int, episode_end: int, episode_at: int):
    n = episode_end - episode_start
    # k = 0.002
    k = 0.002
    x = episode_at - episode_start
    epsilon = pow(k, 2 * x / n)
    return max(epsilon, EPSILON_MIN)


# n_points[n] represents the end of an interval
# multiplier = 7
# n_points = [0, 1800, 1800*multiplier, 1800*multiplier*multiplier*multiplier*multiplier]
# n_points = [0, 1000, 1000*multiplier]

# n_points = [0, 1800, 1800*multiplier, 1800*multiplier*multiplier, 1800*multiplier*multiplier*multiplier]
# n_points = [0, 1000, 2000, 20_000, 40_000]
# n_points = [0, 1000, 2000, 15_000]
# n_points = [0, 1000, 2000, 25_000]

# Constants
# GAMMA = 0.99
# LEARNING_RATE = 0.001/10/10

MODEL_SAVE_PERIOD = 100_000
EPSILON_MIN = 0.00001

# batch_size = 256
# target_update_freq = 1500
# memory_size = 15000
# episodes = n_points[-1]


def init_training(env: RubiksCube, learning_rate: float, memory_size: int):
    input_dim = len(env.flatten())
    output_dim = env.nA
    policy_net = DQN(input_dim, output_dim)
    target_net = DQN(input_dim, output_dim)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    memory = deque(maxlen=memory_size)

    return policy_net, target_net, optimizer, memory


# Function to choose action using epsilon-greedy policy
def select_action(policy: DQN, state, epsilon: float, env: RubiksCube) -> int:
    if random.random() < epsilon:
        return env.sample_action()
    with torch.inference_mode():
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = policy(state)
        return torch.argmax(q_values).item()


# Return n <= max_n, n>=1, with lower chance to return smaller n
def select_n(max_n: int) -> int:
    numbers = list(range(1, max_n + 1))
    weights = list(range(1, max_n + 1))
    return random.choices(numbers, weights=weights, k=1)[0]


def select_n_2(n: int, peak: int = 80) -> int:
    i = np.arange(1, n + 1, dtype=float)
    arr = 3**i
    arr = arr / arr[-1] * peak
    arr[:-1] = arr[:-1] / arr[:-1].sum() * (100 - peak)
    arr[-1] = peak

    return random.choices(range(1, n + 1), weights=arr, k=1)[0]

# Function to optimize the model using experience replay
def optimize_model(
    batch_size: int,
    gamma: float,
    memory: deque,
    policy_net: DQN,
    target_net: DQN,
    optimizer: optim.Optimizer,
):
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


def train(
    gamma: float,
    learning_rate: float,
    batch_size: int,
    target_update_freq: int,
    memory_size: int,
    reward_for_solving: int,
    n_points: list[int],
    create_graph: bool = False,
    save_model: bool = False,
):
    episodes = n_points[-1]
    env = RubiksCube(reward_for_solving=reward_for_solving)
    policy_net, target_net, optimizer, memory = init_training(
        env, learning_rate, memory_size
    )

    if create_graph:
        # Initialize data structures to track rewards and epsilon per selected_n
        max_n = len(n_points) - 1  # Maximum n based on n_points
        rewards_per_n = {
            i: [] for i in range(1, max_n + 1)
        }  # Rewards for each selected_n
        epsilon_per_n = {
            i: [] for i in range(1, max_n + 1)
        }  # Epsilon for each selected_n
        episode_per_n = {
            i: [] for i in range(1, max_n + 1)
        }  # Episode numbers for each selected_n

    # Main training loop
    steps_done = 0
    n = 1

    # Create progress bar for training
    progress_bar = tqdm(range(episodes), desc="Training DQN", unit="episode")

    for episode in progress_bar:
        # Update n and epsilon
        epsilon = calc_epsilon(n_points[n - 1], n_points[n], episode)
        if n < len(n_points) - 1 and episode > n_points[n]:
            n += 1

        selected_n = select_n_2(n)
        state, _ = env.reset(selected_n)
        episode_reward = 0
        done = env.is_solved()
        moves = 0
        max_moves = selected_n

        if done:
            continue

        while not done:
            # Select action
            action = select_action(policy_net, state, epsilon, env)
            moves += 1
            next_state, reward, done, _ = env.step(action, moves, max_moves)

            # Store transition in memory
            memory.append((state, action, reward, next_state, done))

            # Update state
            state = next_state
            episode_reward += reward

            # Optimize model
            optimize_model(batch_size, gamma, memory, policy_net, target_net, optimizer)

            # Update target network periodically
            if steps_done % target_update_freq == 0:
                target_net.load_state_dict(policy_net.state_dict())

            steps_done += 1

        # Update progress bar with current metrics
        progress_bar.set_postfix(
            {
                "epsilon": f"{epsilon:.6f}",
                "reward": f"{episode_reward:.2f}",
                "selected_n": selected_n,
                "memory": len(memory),
            }
        )

        if create_graph:
            # Store rewards, epsilon, and episode for this selected_n
            rewards_per_n[selected_n].append(episode_reward)
            epsilon_per_n[selected_n].append(epsilon)
            episode_per_n[selected_n].append(episode)

        # Save model
        if save_model and (episode % MODEL_SAVE_PERIOD == 0 or episode == episodes - 1):
            torch.save(policy_net.state_dict(), f"rubiks_dqn_policy_net_{episode}.pt")
            torch.save(target_net.state_dict(), f"rubiks_dqn_target_net_{episode}.pt")

    if create_graph:
        # Plotting: Create a single figure with subplots for rewards and epsilon per selected_n
        fig, axes = plt.subplots(max_n, 2, figsize=(12, 4 * max_n), sharex=True)
        if max_n == 1:
            axes = [axes]  # Ensure axes is iterable for single row
        for i in range(max_n):
            selected_n = i + 1
            # Rewards subplot (left column)
            axes[i][0].plot(
                episode_per_n[selected_n],
                rewards_per_n[selected_n],
                color="blue",
                label=f"selected_n = {selected_n}",
            )
            axes[i][0].set_ylabel("Reward")
            axes[i][0].set_title(f"Rewards for selected_n = {selected_n}")
            axes[i][0].legend()
            # Epsilon subplot (right column)
            axes[i][1].plot(
                episode_per_n[selected_n],
                epsilon_per_n[selected_n],
                color="red",
                label=f"selected_n = {selected_n}",
                alpha=0.6,
            )
            axes[i][1].set_ylabel("Epsilon")
            axes[i][1].set_title(f"Epsilon for selected_n = {selected_n}")
            axes[i][1].legend()
        axes[-1][0].set_xlabel("Episode")
        axes[-1][1].set_xlabel("Episode")
        fig.suptitle("Rewards and Epsilon per Episode by selected_n")
        plt.tight_layout()
        plt.show()

    return policy_net
