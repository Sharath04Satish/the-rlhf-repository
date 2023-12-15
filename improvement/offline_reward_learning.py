import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from rollout_policy import (
    get_cumulative_rewards_from_human_demonstrations,
)
from utils import Net
import json
from utils import collect_human_demos


def get_demonstrations_with_returns(env, demos):
    episode, episode_returns = get_cumulative_rewards_from_human_demonstrations(
        env, demos
    )
    return episode, episode_returns


def get_human_demos():
    human_trajectories = list()

    for index in range(5):
        env, demos, _ = collect_human_demos(1, "human", index + 1)
        trajectories, traj_returns = get_demonstrations_with_returns(env, demos)
        human_trajectories.append((trajectories, traj_returns))

    return human_trajectories


def generate_training_data():
    num_iter = 100
    lr = 0.001
    checkpoint = "./reward.params"

    device = "cpu"
    reward_net = Net()
    reward_net.to(device)

    import torch.optim as optim

    optimizer = optim.Adam(reward_net.parameters(), lr=lr)

    human_trajectories = get_human_demos()

    with open("comparisons/data/trajectory_data.json", "r") as data_file:
        trajectory_data = json.load(data_file)
        trajectories = trajectory_data["trajectories"]
        returns = trajectory_data["returns"]

    with open("improvement/data/improvement_indices.json", "r") as data_file:
        random_demonstrations = json.load(data_file)
        random_demonstrations = random_demonstrations["improvement_indices"]

    training_pairs = []
    training_labels = []

    returns_random_demos = [
        value for index, value in enumerate(returns) if index in random_demonstrations
    ]

    returns_human_demos = [value for _, value in human_trajectories]

    improvement_avg_return_condition = np.mean(returns_random_demos) > np.mean(
        returns_human_demos
    )
    improvement_min_max_return_condition = (
        True if max(returns_human_demos) < min(returns_random_demos) else False
    )
    improvement_threshold_condition = (
        True
        if sum(value > max(returns_random_demos) for value in returns_human_demos) < 2
        else False
    )

    if (
        improvement_avg_return_condition
        or improvement_min_max_return_condition
        or improvement_threshold_condition
    ):
        print("Please provide better human demonstrations to improve the model")
        return False
    else:
        improved_human_demonstrations = [
            h_demonstration
            for h_demonstration, h_returns in human_trajectories
            if h_returns > max(returns_random_demos)
        ]

        for index in random_demonstrations:
            traj_i = trajectories[index]
            traj_j = improved_human_demonstrations[
                random.randint(0, len(improved_human_demonstrations) - 1)
            ]

            training_pairs.append((traj_i, traj_j))
            training_labels.append(1)

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
    checkpoint,
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

    torch.save(reward_network.state_dict(), checkpoint)
    print("The reward function has been learnt successfully.")
