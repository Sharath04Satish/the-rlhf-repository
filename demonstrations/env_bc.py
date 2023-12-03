# import gym
import gymnasium as gym
import argparse
import pygame
from teleop import collect_demos
import torch
from torch.optim import Adam
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

device = "cpu"


def collect_human_demos(num_demos):
    mapping = {(pygame.K_LEFT,): 0, (pygame.K_RIGHT,): 2}

    env = gym.make("Acrobot-v1", render_mode="rgb_array")
    state = env.reset()
    demos = collect_demos(env, keys_to_action=mapping, num_demos=num_demos, noop=1)
    env.close()

    return [env, demos]


def torchify_demos(sas_pairs):
    states = []
    actions = []
    next_states = []

    for s, a, s2 in sas_pairs:
        states.append(s)
        actions.append(a)
        next_states.append(s2)

    # Fixing the error called "inhomogeneous element"
    states[0] = states[0][0]
    next_states.remove(next_states[0])

    states = np.array(states)
    actions = np.array(actions)
    next_states = np.array(next_states)

    obs_torch = torch.from_numpy(np.array(states)).float().to(device)
    obs2_torch = torch.from_numpy(np.array(next_states)).float().to(device)
    acs_torch = torch.from_numpy(np.array(actions)).float().to(device)

    return obs_torch, acs_torch, obs2_torch


def visualize_trajectory(env, obs):
    env.reset()
    for state in obs:
        env.render()
        env.step(state)


def sample_comparisons(demos, num_comparisons=2):
    comparison_indices = np.random.choice(
        len(demos), size=num_comparisons, replace=False
    )
    comparisons = [demos[i] for i in comparison_indices]
    return comparisons


def train_policy(obs, acs, nn_policy, num_train_iters):
    pi_optimizer = Adam(nn_policy.parameters(), lr=0.1)
    # action space is discrete so our policy just needs to classify which action to take
    # we typically train classifiers using a cross entropy loss
    loss_criterion = nn.CrossEntropyLoss()

    # run BC using all the demos in one giant batch
    for i in range(num_train_iters):
        # zero out automatic differentiation from last time
        pi_optimizer.zero_grad()
        # run each state in batch through policy to get predicted logits for classifying action
        pred_action_logits = nn_policy(obs)
        # print(pred_action_logits)
        _, pred_actions = pred_action_logits.max(dim=1)
        pred_actions = torch.tensor(pred_actions, dtype=torch.float32)
        pred_action_logits = torch.tensor(
            pred_action_logits.detach().numpy().argmax(axis=1), dtype=torch.float32
        )
        # print(pred_actions)
        # now compute loss by comparing what the policy thinks it should do with what the demonstrator didd
        loss = loss_criterion(pred_actions, acs)
        # back propagate the error through the network to figure out how update it to prefer demonstrator actions
        loss.requires_grad = True
        loss.backward()
        # perform update on policy parameters
        pi_optimizer.step()


class PolicyNetwork(nn.Module):
    """
    Simple neural network with two layers that maps a 5-d state to a prediction
    over which of the two discrete actions should be taken.
    The two outputs corresponding to the logits for a 2-way classification problem.

    """

    def __init__(self):
        super().__init__()
        # TODO define network architecture.
        # Hint, states in cartpole are 4-dimensional (x,xdot,theta,thetadot)
        # https://www.gymlibrary.dev/environments/classic_control/cart_pole/

        # Define the number of nodes in each layer of the neural network.
        # This neural network has an input layer, two hidden layers and an output layer.
        num_nodes_in_input_layer = 6
        num_nodes_in_hidden_layer_1 = 64
        num_nodes_in_hidden_layer_2 = 16
        num_nodes_in_output_layer = 3

        # Input layer to the first hidden layer.
        self.linear1 = nn.Linear(num_nodes_in_input_layer, num_nodes_in_hidden_layer_1)
        self.activation1 = nn.ReLU()

        # First hidden layer to the second hidden layer.
        self.linear2 = nn.Linear(
            num_nodes_in_hidden_layer_1, num_nodes_in_hidden_layer_2
        )
        self.activation2 = nn.ReLU()

        # Second hidden layer to the output layer.
        self.output = nn.Linear(num_nodes_in_hidden_layer_2, num_nodes_in_output_layer)
        self.activation3 = nn.Softmax()

    def forward(self, input_observation):
        input_observation = self.activation1(self.linear1(input_observation))
        input_observation = self.activation2(self.linear2(input_observation))
        input_observation = self.activation3(self.output(input_observation))

        return input_observation


# evaluate learned policy
def evaluate_policy(pi, num_evals, human_render=True):
    if human_render:
        env = gym.make("Acrobot-v1", render_mode="rgb_array")
        env.reset()
    else:
        env = gym.make("Acrobot-v1", render_mode="rgb_array")
        env.reset()

    print(pi)
    policy_returns = []
    for i in tqdm(range(num_evals)):
        done = False
        total_reward = 0
        obs = env.reset()[0]
        # print(obs)
        max_count = 1
        while max_count < 100:
            # take the action that the network assigns the highest logit value to
            # Note that first we convert from numpy to tensor and then we get the value of the
            # argmax using .item() and feed that into the environment
            print(pi(torch.from_numpy(obs).unsqueeze(0)))
            action = torch.argmax(pi(torch.from_numpy(obs).unsqueeze(0))).item()
            # print(action)

            obs, rew, done, _, _ = env.step(action)
            total_reward += rew
            max_count += 1
        print("reward for evaluation", i, total_reward)
        policy_returns.append(total_reward)

    print("average policy return", np.mean(policy_returns))
    print("min policy return", np.min(policy_returns))
    print("max policy return", np.max(policy_returns))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument(
        "--num_demos",
        default=3,
        type=int,
        help="number of rgb_array demonstrations to collect",
    )
    parser.add_argument(
        "--num_bc_iters", default=100, type=int, help="number of iterations to run BC"
    )
    parser.add_argument(
        "--num_evals",
        default=6,
        type=int,
        help="number of times to run policy after training for evaluation",
    )

    args = parser.parse_args()

    # collect rgb_array demos
    env, demos = collect_human_demos(args.num_demos)

    # process demos
    obs, acs, _ = torchify_demos(demos)

    # print(env.action_space, env.observation_space)

    input_layer_size = 6
    hidden_layer_size = 64
    output_layer_size = 3

    # train policy
    pi = PolicyNetwork()
    train_policy(obs, acs, pi, args.num_bc_iters)

    print("Policy has been trained. Evaluating policy.")

    # evaluate learned policy
    evaluate_policy(pi, args.num_evals)
