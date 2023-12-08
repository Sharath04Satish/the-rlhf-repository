import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np


def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    # Build a feedforward neural network for the policy
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


class Net(nn.Module):
    ##TODO you can code this up using whatever methods you want,
    # but to work with the rest of the code make sure you
    # at least implement the predict_reward function below

    def __init__(self):
        super().__init__()
        # TODO define network architecture.
        # Hint, states in cartpole are 4-dimensional (x,xdot,theta,thetadot)
        # https://www.gymlibrary.dev/environments/classic_control/cart_pole/

        # Define the number of nodes in each layer of the neural network.
        # This neural network has an input layer, two hidden layers and an output layer.
        num_nodes_in_input_layer = 4
        num_nodes_in_hidden_layer_1 = 64
        num_nodes_in_hidden_layer_2 = 16
        num_nodes_in_output_layer = 1

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
        self.activation5 = nn.Sigmoid()

    def forward(self, input_observation):
        input_observation = self.activation1(self.linear1(input_observation))
        input_observation = self.activation2(self.linear2(input_observation))
        input_observation = self.activation5(self.output(input_observation))

        return input_observation

    def predict_reward(self, traj):
        """calculate cumulative return of trajectory, could be a trajectory with a single element"""
        # TODO should take in a trajectory and output a scalar cumulative reward estimate

        return_value = 0
        for trajectory in traj:
            trajectory = torch.tensor(trajectory).float()
            trajectory_reward = self.forward(trajectory)
            return_value += trajectory_reward

        return return_value
