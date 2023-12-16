import gym
import torch
import torch.nn as nn
import numpy as np
from rollout_policy_syn import (
    generate_rollout_1,
)
from utils_syn import mlp, Net
import json


def generate_novice_demos(env):
    checkpoints = []
    for i in range(10):
        checkpoints.append("synthetic/policy_checkpoint" + str(i) + ".params")

    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env.reset()
    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n
    hidden_sizes = [32]
    device = "cpu"

    demonstrations = []
    demo_returns = []

    for index, checkpoint in enumerate(checkpoints):
        policy = mlp(sizes=[obs_dim] + hidden_sizes + [n_acts])
        policy.load_state_dict(torch.load(checkpoint))
        traj, ret = generate_rollout_1(policy, env, index + 1)

        demonstrations.append(traj)
        demo_returns.append(ret)

    return demonstrations, demo_returns


def get_store_novice_demonstrations():
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env.reset()

    trajectories, traj_returns = generate_novice_demos(env)

    json_data = {
        "trajectories": trajectories,
        "returns": traj_returns,
    }

    with open("comparisons/data/trajectory_data.json", "w") as data_file:
        json.dump(json_data, data_file)


def generate_training_data():
    num_iter = 100
    lr = 0.001
    checkpoint = "./reward.params"

    device = "cpu"
    reward_net = Net()
    reward_net.to(device)

    import torch.optim as optim

    optimizer = optim.Adam(reward_net.parameters(), lr=lr)

    with open("comparisons/data/trajectory_data.json", "r") as data_file:
        trajectory_data = json.load(data_file)
        trajectories = trajectory_data["trajectories"]

    with open("comparisons/data/comparisons_preferences.json", "r") as data_file:
        human_preference_choices = json.load(data_file)
        human_preferences = human_preference_choices["preferences"]
        human_preferences = [
            (int(x) - 1, int(y) - 1)
            for x, y in [item.strip("()").split(",") for item in human_preferences]
        ]

    training_pairs = []
    training_labels = []

    for demonstration_a, demonstration_b in human_preferences:
        ti = demonstration_a
        tj = demonstration_b

        traj_i = trajectories[ti]
        traj_j = trajectories[tj]

        if int(demonstration_a) == ti:
            label = 0
        else:
            label = 1

        training_pairs.append((traj_i, traj_j))
        training_labels.append(label)

    return [
        reward_net,
        optimizer,
        training_pairs,
        training_labels,
        num_iter,
        checkpoint,
    ]


def learn_reward_function(
    reward_network,
    optimizer,
    training_pairs,
    training_labels,
    num_iter,
    checkpoint_dir,
):
    device = "cpu"
    loss_criterion = nn.CrossEntropyLoss()

    for _ in range(num_iter):
        total_loss = 0.0
        for index in range(len(training_pairs)):
            optimizer.zero_grad()
            traj_i, traj_j = training_pairs[index]
            label = torch.tensor([training_labels[index]], dtype=torch.float32).to(
                device
            )

            traj_i = torch.tensor(np.array(traj_i), dtype=torch.float32).to(device)
            traj_j = torch.tensor(np.array(traj_j), dtype=torch.float32).to(device)

            reward_i = reward_network.predict_reward(traj_i)
            reward_j = reward_network.predict_reward(traj_j)

            pred_logit = 1
            if reward_i > reward_j:
                pred_logit = 0

            logits = torch.tensor([pred_logit], dtype=torch.float32)

            loss = loss_criterion(logits, label)
            total_loss += loss.item()
            loss.requires_grad = True
            loss.backward()
            optimizer.step()

    torch.save(reward_network.state_dict(), checkpoint_dir)
    print("The reward function has been learnt successfully.")
