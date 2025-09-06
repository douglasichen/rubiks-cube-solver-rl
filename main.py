from dqn import train
from test import test

policy = train(
    gamma=0.99,
    learning_rate=0.00001,
    batch_size=256,
    target_update_freq=1500,
    memory_size=15000,
    reward_for_solving=3,
    save_model=True,
    n_points=[0, 1, 2, 10000, 50000],
    create_graph=True,
)


for n in range(14):
    print(f"n={n}: {test(policy, n, 100) * 100:.1f}% success rate")
