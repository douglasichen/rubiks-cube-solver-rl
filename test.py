import torch
from env import RubiksCube
from dqn import DQN, select_action


def test(policy_net: DQN, n: int, num_tests: int):
    env = RubiksCube(reward_for_solving=3)
    num_solved = 0
    for _ in range(num_tests):
        state, _ = env.reset(n)
        done = env.is_solved()
        moves = 0
        rewards = 0
        while not done:
            action = select_action(policy_net, state, 0, env)
            moves += 1
            state, reward, done, _ = env.step(action, moves, n)  # FIXED: Update state!
            rewards += reward
        if done and rewards > -n:
            num_solved += 1

    return num_solved / num_tests


policy_net_file = "pretrained_model.pt"
policy_net = DQN(144, 6)
state_dict = torch.load(policy_net_file, map_location="cpu", weights_only=True)
policy_net.load_state_dict(state_dict)
policy_net.eval()

for n in range(1, 11):
    print(f"n={n}: {test(policy_net, n, 1000) * 100:.5f}% success rate")