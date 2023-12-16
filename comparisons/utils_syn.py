import torch
import torch.nn as nn


def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        num_nodes_in_input_layer = 4
        num_nodes_in_hidden_layer_1 = 64
        num_nodes_in_hidden_layer_2 = 16
        num_nodes_in_output_layer = 1

        self.linear1 = nn.Linear(num_nodes_in_input_layer, num_nodes_in_hidden_layer_1)
        self.activation1 = nn.ReLU()
        self.linear2 = nn.Linear(
            num_nodes_in_hidden_layer_1, num_nodes_in_hidden_layer_2
        )
        self.activation2 = nn.ReLU()
        self.output = nn.Linear(num_nodes_in_hidden_layer_2, num_nodes_in_output_layer)
        self.activation5 = nn.Sigmoid()

    def forward(self, input_observation):
        input_observation = self.activation1(self.linear1(input_observation))
        input_observation = self.activation2(self.linear2(input_observation))
        input_observation = self.activation5(self.output(input_observation))

        return input_observation

    def predict_reward(self, traj):
        """calculate cumulative return of trajectory, could be a trajectory with a single element"""

        return_value = 0
        for trajectory in traj:
            trajectory = torch.tensor(trajectory).float()
            trajectory_reward = self.forward(trajectory)
            return_value += trajectory_reward

        return return_value
