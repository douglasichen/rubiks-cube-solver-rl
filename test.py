import torch
from env import RubiksCube
from dqn import DQN, select_action


def test(policy_net: DQN, n: int, num_tests: int):
    env = RubiksCube(reward_for_solving=3)
    num_solved = 0
    for _ in range(num_tests):
        state, _ = env.reset(n)
        done = False
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


# policy_net_file = "models/rubiks_dqn_policy_net_24999.pt"
# policy_net = DQN(144, 6)
# state_dict = torch.load(policy_net_file, map_location="cpu", weights_only=True)
# policy_net.load_state_dict(state_dict)
# policy_net.eval()
# print("n=1:", test(policy_net, 1, 1000)*100, "% success rate")
# print("n=2:", test(policy_net, 2, 1000)*100, "% success rate")
# print("n=3:", test(policy_net, 3, 1000)*100, "% success rate") 
# print("n=4:", test(policy_net, 4, 1000)*100, "% success rate") 
# print("n=5:", test(policy_net, 5, 1000)*100, "% success rate") 
# print("n=6:", test(policy_net, 6, 1000)*100, "% success rate")
# print("n=7:", test(policy_net, 7, 1000)*100, "% success rate")
# print("n=8:", test(policy_net, 8, 1000)*100, "% success rate") 
# print("n=9:", test(policy_net, 9, 1000)*100, "% success rate") 
# print("n=10:", test(policy_net, 10, 1000)*100, "% success rate") 
# print("n=11:", test(policy_net, 11, 1000)*100, "% success rate")
# print("n=12:", test(policy_net, 12, 1000)*100, "% success rate")
# print("n=13:", test(policy_net, 13, 1000)*100, "% success rate") 
# print("n=14:", test(policy_net, 14, 1000)*100, "% success rate")
