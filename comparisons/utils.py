import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import pygame
import gymnasium as gym
import gym.error
from gym import Env, logger
from typing import Callable, Dict, List, Optional, Tuple, Union
from gym.wrappers.monitoring.video_recorder import VideoRecorder

try:
    import pygame
    from pygame import Surface
    from pygame.event import Event
    from pygame.locals import VIDEORESIZE
except ImportError:
    raise gym.error.DependencyNotInstalled(
        "Pygame is not installed, run `pip install gym[classic_control]`"
    )


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
        num_nodes_in_input_layer = 2
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
        self.activation3 = nn.Sigmoid()

    def forward(self, input_observation):
        input_observation = self.activation1(self.linear1(input_observation))
        input_observation = self.activation2(self.linear2(input_observation))
        input_observation = self.activation3(self.output(input_observation))

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


def collect_human_demos(num_demos, demo_type, index):
    mapping = {(pygame.K_LEFT,): 0, (pygame.K_RIGHT,): 2}

    env = gym.make("MountainCar-v0", render_mode="rgb_array")
    state = env.reset()
    demos = collect_demos(
        env,
        demo_type=demo_type,
        index=index,
        keys_to_action=mapping,
        num_demos=num_demos,
        noop=1,
    )

    env.close()

    return [env, demos]


def collect_demos(
    env: Env,
    demo_type,
    index,
    transpose: Optional[bool] = True,
    fps: Optional[int] = None,
    zoom: Optional[float] = None,
    callback: Optional[Callable] = None,
    keys_to_action=None,
    seed: Optional[int] = None,
    noop=1,
    num_demos: Optional[int] = 1,
):
    obs = env.reset()

    if keys_to_action is None:
        if hasattr(env, "get_keys_to_action"):
            keys_to_action = env.get_keys_to_action()
        elif hasattr(env.unwrapped, "get_keys_to_action"):
            keys_to_action = env.unwrapped.get_keys_to_action()
    assert keys_to_action is not None

    key_code_to_action = {}
    for key_combination, action in keys_to_action.items():
        key_code = tuple(
            sorted(ord(key) if isinstance(key, str) else key for key in key_combination)
        )
        key_code_to_action[key_code] = action

    game = PlayableGame(env, key_code_to_action, zoom)

    if fps is None:
        fps = env.metadata.get("render_fps", 30)

    done = False

    clock = pygame.time.Clock()
    steps = 0
    episodes = 0
    total_reward = 0
    sas_pairs = []
    video = VideoRecorder(
        env, "comparisons_data/{0}_demonstration_{1}.mp4".format(demo_type, index)
    )
    while episodes < num_demos:
        if done:
            done = False
            print("total reward", total_reward)
            total_reward = 0
            episodes += 1
            obs = env.reset()

        else:
            steps += 1
            action = key_code_to_action.get(tuple(sorted(game.pressed_keys)), noop)
            prev_obs = obs
            video.capture_frame()
            obs, rew, done, _, info = env.step(action)
            sas_pairs.append((prev_obs, action, obs))
            total_reward += rew
            if callback is not None:
                callback(prev_obs, obs, action, rew, done, info)
        if obs is not None:
            rendered = env.render()
            if isinstance(rendered, List):
                rendered = rendered[-1]
            assert rendered is not None and isinstance(rendered, np.ndarray)
            display_arr(
                game.screen, rendered, transpose=transpose, video_size=game.video_size
            )

        # process pygame events
        for event in pygame.event.get():
            game.process_event(event)

        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()
    video.close()
    env.close()

    return sas_pairs


def display_arr(
    screen: Surface, arr: np.ndarray, video_size: Tuple[int, int], transpose: bool
):
    arr_min, arr_max = np.min(arr), np.max(arr)
    arr = 255.0 * (arr - arr_min) / (arr_max - arr_min)
    pyg_img = pygame.surfarray.make_surface(arr.swapaxes(0, 1) if transpose else arr)
    pyg_img = pygame.transform.scale(pyg_img, video_size)
    screen.blit(pyg_img, (0, 0))


class PlayableGame:
    """Wraps an environment allowing keyboard inputs to interact with the environment."""

    def __init__(
        self,
        env: Env,
        keys_to_action: Optional[Dict[Tuple[int, ...], int]] = None,
        zoom: Optional[float] = None,
    ):
        self.env = env
        self.relevant_keys = self._get_relevant_keys(keys_to_action)
        self.video_size = self._get_video_size(zoom)
        self.screen = pygame.display.set_mode(self.video_size)
        self.pressed_keys = []
        self.running = True

    def _get_relevant_keys(
        self, keys_to_action: Optional[Dict[Tuple[int], int]] = None
    ) -> set:
        if keys_to_action is None:
            if hasattr(self.env, "get_keys_to_action"):
                keys_to_action = self.env.get_keys_to_action()
            elif hasattr(self.env.unwrapped, "get_keys_to_action"):
                keys_to_action = self.env.unwrapped.get_keys_to_action()
        assert isinstance(keys_to_action, dict)
        relevant_keys = set(sum((list(k) for k in keys_to_action.keys()), []))
        return relevant_keys

    def _get_video_size(self, zoom: Optional[float] = None) -> Tuple[int, int]:
        rendered = self.env.render()
        print(rendered)
        if isinstance(rendered, List):
            rendered = rendered[-1]
        assert rendered is not None and isinstance(rendered, np.ndarray)
        video_size = [rendered.shape[1], rendered.shape[0]]

        if zoom is not None:
            video_size = int(video_size[0] * zoom), int(video_size[1] * zoom)

        return video_size

    def process_event(self, event: Event):
        """Processes a PyGame event.

        In particular, this function is used to keep track of which buttons are currently pressed
        and to exit the :func:`play` function when the PyGame window is closed.

        Args:
            event: The event to process
        """
        if event.type == pygame.KEYDOWN:
            if event.key in self.relevant_keys:
                self.pressed_keys.append(event.key)
            elif event.key == pygame.K_ESCAPE:
                self.running = False
        elif event.type == pygame.KEYUP:
            if event.key in self.relevant_keys:
                self.pressed_keys.remove(event.key)
        elif event.type == pygame.K_w:
            if event.key in self.relevant_keys:
                self.pressed_keys.remove(event.key)
        elif event.type == pygame.K_a:
            if event.key in self.relevant_keys:
                self.pressed_keys.remove(event.key)
        elif event.type == pygame.K_d:
            if event.key in self.relevant_keys:
                self.pressed_keys.remove(event.key)
        elif event.type == pygame.QUIT:
            self.running = False
        elif event.type == VIDEORESIZE:
            self.video_size = event.size
            self.screen = pygame.display.set_mode(self.video_size)
