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
    get_cumulative_rewards_from_human_demonstrations,
)
from utils import mlp, Net, collect_human_demos
from constants import (
    num_human_demonstrations_for_comparisons,
    num_synthetic_demonstrations_for_comparisons,
)


def get_demonstrations_with_returns(env, demos):
    episode, episode_returns = get_cumulative_rewards_from_human_demonstrations(
        env, demos
    )
    return episode, episode_returns


def generate_novice_demos(env, demos):
    checkpoints = []
    for i in range(10):
        checkpoints.append("./synthetic/policy_checkpoint" + str(i) + ".params")

    # make core of policy network
    # env = gym.make("MountainCar-v0")
    env = gym.make("MountainCar-v0", render_mode="rgb_array")
    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n
    hidden_sizes = [32]
    device = "cpu"

    demonstrations = []
    demo_returns = []

    for checkpoint in checkpoints:
        policy = mlp(sizes=[obs_dim] + hidden_sizes + [n_acts])
        # policy.load_state_dict(torch.load(checkpoint))
        traj, ret = generate_rollout(None, env, demos)
        demonstrations.append(traj)
        demo_returns.append(ret)

    return demonstrations, demo_returns


def create_training_data_human(human_demonstrations):
    training_pairs = []
    training_labels = []

    for _ in range(len(human_demonstrations)):
        ti = 0
        tj = 0

        # only add trajectories that are different returns
        while ti == tj:
            # pick two random demonstrations
            ti = np.random.randint(len(human_demonstrations))
            tj = np.random.randint(len(human_demonstrations))
        # create random partial trajs by finding random start frame and random skip frame

        traj_i = human_demonstrations[ti][0]
        traj_j = human_demonstrations[tj][0]

        if human_demonstrations[ti][1] > human_demonstrations[tj][1]:
            label = 0
        else:
            label = 1

        training_pairs.append((traj_i, traj_j))
        training_labels.append(label)

    return training_pairs, training_labels


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

        if cum_returns[ti] > cum_returns[tj]:
            label = 0
        else:
            label = 1

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
    torch.save(reward_net.state_dict(), checkpoint_dir)
    print("finished training")


if __name__ == "__main__":
    human_trajectories = list()
    for index in range(num_human_demonstrations_for_comparisons):
        env, demos = collect_human_demos(1, "human", index + 1)
        trajectories, traj_returns = get_demonstrations_with_returns(env, demos)
        human_trajectories.append((trajectories, traj_returns))

    traj_pairs, traj_labels = create_training_data_human(human_trajectories)

    num_iter = 100
    lr = 0.001
    checkpoint = "./reward.params"  # where to save your reward function weights

    device = "cpu"
    reward_net = Net()
    reward_net.to(device)

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
