import gym
import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import random
from rollout_policy import (
    generate_rollout,
    generate_rollout_ppo_sgd,
    generate_rollout_1,
    get_cumulative_rewards_from_human_demonstrations,
)
from utils import mlp, Net
from random import choice
import json
from utils import collect_human_demos
from constants import threshold_improved_demos


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


def generate_novice_demos(env):
    checkpoints = []
    for i in range(10):
        checkpoints.append("synthetic/policy_checkpoint" + str(i) + ".params")

    # make core of policy network
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
        # traj, ret = generate_rollout_ppo_sgd(index + 1)
        print("traj ground-truth return", ret)
        demonstrations.append(traj)
        demo_returns.append(ret)

    return demonstrations, demo_returns


def create_training_data(trajectories, cum_returns, num_pairs):
    training_pairs = []
    training_labels = []
    num_trajs = len(trajectories)

    # add pairwise preferences over full trajectories
    for n in range(num_trajs):
        ti = 0
        tj = 0
        # only add trajectories that are different returns
        while ti == tj:
            # pick two random demonstrations
            ti = np.random.randint(num_trajs)
            tj = np.random.randint(num_trajs)
        # create random partial trajs by finding random start frame and random skip frame

        traj_i = trajectories[ti]
        traj_j = trajectories[tj]

        comparison_selection = choice([ti, tj])

        if comparison_selection == ti:
            label = 0
        else:
            label = 1
        # print(cum_returns[ti], cum_returns[tj])
        # print(label)

        training_pairs.append((traj_i, traj_j))
        training_labels.append(label)

    return training_pairs, training_labels


def predict_traj_return(net, traj):
    traj = np.array(traj)
    traj = torch.from_numpy(traj).float().to(device)
    return net.predict_reward(traj).item()


# Train the network
def learn_reward(
    reward_network,
    optimizer,
    training_inputs,
    training_outputs,
    num_iter,
    checkpoint_dir,
):
    # check if gpu available
    device = "cpu"

    # We will use a cross entropy loss for pairwise preference learning
    loss_criterion = nn.CrossEntropyLoss()

    # TODO: train reward function using the training data
    # training_inputs gives you a list of pairs of trajectories
    # training_outputs gives you a list of labels (0 if first trajectory better, 1 if second is better)
    for _ in range(num_iter):
        total_loss = 0.0
        for index in range(len(training_inputs)):
            optimizer.zero_grad()
            traj_i, traj_j = training_inputs[index]
            label = torch.tensor([training_outputs[index]], dtype=torch.float32).to(
                device
            )

            # Convert the trajectories to tensors
            traj_i = torch.tensor(np.array(traj_i), dtype=torch.float32).to(device)
            traj_j = torch.tensor(np.array(traj_j), dtype=torch.float32).to(device)

            # Predict cumulative rewards for each trajectory
            reward_i = reward_network.predict_reward(traj_i)
            reward_j = reward_network.predict_reward(traj_j)

            pred_logit = 1
            if reward_i > reward_j:
                pred_logit = 0

            # Concatenate the rewards to form logits
            logits = torch.tensor([pred_logit], dtype=torch.float32)

            # Calculate the loss
            loss = loss_criterion(logits, label)
            total_loss += loss.item()

            # Backpropagation
            loss.requires_grad = True
            loss.backward()
            optimizer.step()

    # After training we save the reward function weights
    print("check pointing")
    torch.save(reward_network.state_dict(), checkpoint_dir)
    print("finished training")


def get_store_novice_demonstrations():
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env.reset()

    trajectories, traj_returns = generate_novice_demos(env)

    json_data = {
        "trajectories": trajectories,
        "returns": traj_returns,
    }

    with open("../comparisons/data/trajectory_data.json", "w") as data_file:
        json.dump(json_data, data_file)


def generate_training_data():
    # TODO: hyper parameters that you may want to tweak or change
    num_iter = 100
    lr = 0.001
    checkpoint = "./reward.params"  # where to save your reward function weights

    # Now we create a reward network and optimize it using the training data.
    # TODO: You will need to code up Net in utils.py
    device = "cpu"
    reward_net = Net()
    reward_net.to(device)

    print("Reward nets", reward_net)

    import torch.optim as optim

    optimizer = optim.Adam(reward_net.parameters(), lr=lr)

    human_trajectories = get_human_demos()

    with open("../comparisons/data/trajectory_data.json", "r") as data_file:
        trajectory_data = json.load(data_file)
        trajectories = trajectory_data["trajectories"]
        returns = trajectory_data["returns"]

    with open("data/improvement_indices.json", "r") as data_file:
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
        # add pairwise preferences over full trajectories
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
    reward_net,
    optimizer,
    training_pairs,
    training_labels,
    num_iter,
    checkpoint,
):
    learn_reward(
        reward_net,
        optimizer,
        training_pairs,
        training_labels,
        num_iter,
        checkpoint,
    )


if __name__ == "__main__":
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env.reset()

    num_pairs = 20
    # create synthetic trajectories for RLHF
    trajectories, traj_returns = generate_novice_demos(env)

    # create pairwise preference data using ground-truth reward
    traj_pairs, traj_labels = create_training_data(
        trajectories, traj_returns, num_pairs
    )

    # TODO: hyper parameters that you may want to tweak or change
    num_iter = 100
    lr = 0.001
    checkpoint = "./reward.params"  # where to save your reward function weights

    # Now we create a reward network and optimize it using the training data.
    # TODO: You will need to code up Net in utils.py
    device = "cpu"
    reward_net = Net()
    reward_net.to(device)

    print("Reward nets", reward_net)

    import torch.optim as optim

    optimizer = optim.Adam(reward_net.parameters(), lr=lr)

    # TODO: You will need to implement learn_reward, you can add arguments or do whatever you want
    learn_reward(
        reward_net,
        optimizer,
        traj_pairs,
        traj_labels,
        num_iter,
        checkpoint,
    )

    # debugging printout
    # we should see higher predicted rewards for more preferred trajectories
    print("performance on training data")
    for i, pair in enumerate(traj_pairs):
        trajA, trajB = pair
        print("predicted return trajA", predict_traj_return(reward_net, trajA))
        print("predicted return trajB", predict_traj_return(reward_net, trajB))
        if traj_labels[i] == 0:
            print("A should be better")
        else:
            print("B should be better")
