# Rubik's Cube Solver using Deep Q-Network (DQN)

A reinforcement learning project that trains a Deep Q-Network to solve a 2x2x2 Rubik's cube using PyTorch.

## Overview

This project implements a Deep Q-Network (DQN) agent that learns to solve a Rubik's cube through reinforcement learning. The cube is represented as a 2x2x2 configuration with 6 faces, and the agent learns to perform the optimal sequence of moves to return the cube to its solved state.

## Algorithm Approach

Similar to how humans learn, neural networks perform better when trained on incrementally increasing difficulty levels. This project implements a curriculum learning approach where the agent starts with simple cube configurations and gradually progresses to more complex ones.

### Key Algorithm Components

**Progressive Difficulty Training**: The algorithm controls an independent variable `n` representing the number of random moves applied to scramble the cube from its solved state. As `n` increases, the problem becomes more challenging, allowing the agent to build up its solving capabilities step by step.

**DQN Framework Requirements**:
- **Environment**: Computes the next cube state given an action
- **State Representation**: Binary encoding (1s and 0s) for each cube face to help the network quickly understand the relevance of different cube colors
- **Actions**: 6 discrete actions corresponding to different cube rotations
- **Reward Function**: 
  - +5 for reaching the solved state (encourages success)
  - -1 for each move taken (encourages efficiency and fewer moves)

## Features

- **Custom Rubik's Cube Environment**: A simplified 2x2x2 cube implementation with 6 possible actions
- **Deep Q-Network**: Neural network architecture for Q-value approximation
- **Experience Replay**: Memory buffer for storing and sampling past experiences
- **Target Network**: Separate target network for stable Q-learning
- **Adaptive Training**: Progressive difficulty with different cube scrambling levels
- **Visualization**: Training progress plots showing rewards and epsilon values
- **Model Checkpointing**: Automatic saving of trained models

## Project Structure

```
rubiks-cube-solver-rl/
├── rubiks_dqn.py          # Main training script with DQN implementation
├── env.py                 # Rubik's cube environment implementation
├── sandbox.py             # Testing and debugging script
├── requirements.txt       # Python dependencies
├── README.md             # This documentation
└── *.pth                 # Saved model checkpoints
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/douglasichen/rubiks-cube-solver-rl.git
cd rubiks-cube-solver-rl
```

2. Create and activate a virtual environment:
```bash
python -m venv local.venv
source local.venv/bin/activate  # On Windows: local.venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

Run the main training script:
```bash
python rubiks_dqn.py
```

The training process will:
- Train for 20,000 episodes by default (configurable via n_points)
- Save model checkpoints every 100,000 episodes
- Display training progress plots at the end
- Use adaptive difficulty levels (1-3 moves to scramble by default)

### Testing the Environment

Use the sandbox script to test the cube environment:
```bash
python sandbox.py
```

### Loading Pre-trained Models

To load a pre-trained model for evaluation or continued training:

```python
import torch
from rubiks_dqn import DQN

# Load a pre-trained model
model = DQN(input_dim=144, output_dim=6)
model.load_state_dict(torch.load('rubiks_dqn_policy_net_88199.pth'))
model.eval()  # Set to evaluation mode
```

## Environment Details

### State Representation
- **State Space**: 144-dimensional flattened vector representing the cube configuration
- **Binary Encoding**: Each cube face is represented using only 1s and 0s to help the network quickly understand the relevance of different cube colors
- **Action Space**: 6 discrete actions corresponding to different cube rotations
- **Reward Structure**:
  - +5 for solving the cube (encourages success)
  - -1 for each move taken (encourages efficiency and fewer moves)

### Actions
The cube supports 6 different rotation actions:
- Action 0: Step 1 rotation
- Action 1: Step 2 rotation  
- Action 2: Step 3 rotation
- Action 3: Step 4 rotation
- Action 4: Step 5 rotation
- Action 5: Step 6 rotation

## Model Architecture

### DQN Network
- **Input Layer**: 144 neurons (flattened cube state)
- **Hidden Layer 1**: 256 neurons with ReLU activation
- **Hidden Layer 2**: 256 neurons with ReLU activation
- **Output Layer**: 6 neurons (Q-values for each action)

### Training Parameters
- **Learning Rate**: 0.00001
- **Gamma (Discount Factor)**: 0.99
- **Epsilon Decay**: Exponential decay with k=0.002 and minimum of 0.00001
- **Batch Size**: 256
- **Memory Size**: 15,000 experiences
- **Target Network Update**: Every 1,500 steps
- **Optimizer**: Adam optimizer

## Training Strategy

The training uses a progressive difficulty approach:

1. **Phase 1** (Episodes 0-1,000): Learn to solve cubes scrambled with 1 move
2. **Phase 2** (Episodes 1,000-2,000): Learn to solve cubes scrambled with 2 moves
3. **Phase 3** (Episodes 2,000-20,000): Learn to solve cubes scrambled with 3 moves

Each phase uses adaptive epsilon-greedy exploration with exponential decay. Within each phase, the difficulty level (number of scrambling moves) is selected using weighted random selection, favoring higher difficulty levels.

## Results and Visualization

The training script generates plots showing:
- Episode rewards for each difficulty level
- Epsilon values over time for each phase
- Training progress across different scrambling levels

## Model Checkpoints

Trained models are automatically saved as:
- `rubiks_dqn_policy_net_{episode}.pth`: Policy network weights
- `rubiks_dqn_target_net_{episode}.pth`: Target network weights

### Available Pre-trained Models

The repository includes several pre-trained model checkpoints:
- Models saved at episodes: 0, 9999, 11999, 12999, 13999, 14999, 17999, 19999, 88199
- Each checkpoint includes both policy and target network weights
- The highest episode model (88199) represents the most trained version

## Dependencies

- **PyTorch**: Deep learning framework
- **NumPy**: Numerical computations
- **Matplotlib**: Plotting and visualization
- **Collections**: Deque for experience replay buffer

## Customization

### Modifying Training Parameters
Edit the constants in `rubiks_dqn.py`:
```python
MODEL_SAVE_PERIOD = 100_000    # Save frequency
GAMMA = 0.99                   # Discount factor
LEARNING_RATE = 0.001/10/10    # Learning rate
EPSILON_MIN = 0.00001         # Minimum exploration
```

### Adjusting Training Phases
Modify the `n_points` list to change training phases:
```python
n_points = [0, 1000, 2000, 20_000]
```

### Changing Network Architecture
Modify the DQN class in `rubiks_dqn.py`:
```python
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)  # Hidden layer size
        self.fc2 = nn.Linear(256, 256)        # Hidden layer size
        self.fc3 = nn.Linear(256, output_dim) # Output layer
```

## Troubleshooting

### Common Issues

1. **CUDA/GPU Issues**: The code uses CPU by default. To use GPU, modify the tensor creation:
   ```python
   state = torch.FloatTensor(state).unsqueeze(0).cuda()
   ```

2. **Memory Issues**: Reduce batch size or memory size if running out of RAM:
   ```python
   initial_batch_size = 128
   initial_memory_size = 10000
   ```

3. **Training Too Slow**: Reduce the number of episodes or simplify the network architecture.

## Future Improvements

- [ ] Add GPU support for faster training
- [ ] Implement Double DQN for better stability
- [ ] Add curriculum learning with more sophisticated difficulty progression
- [ ] Implement 3x3x3 cube support
- [ ] Add visualization of cube solving process
- [ ] Implement model evaluation and testing scripts

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## Development Status

This project is currently on the `feat/modularize` branch, which includes:
- Modularized code structure
- Pre-trained model checkpoints
- Improved training configuration
- Enhanced documentation

## Acknowledgments

This project is inspired by the classic Deep Q-Network paper and the challenge of applying reinforcement learning to combinatorial puzzles like the Rubik's cube.
