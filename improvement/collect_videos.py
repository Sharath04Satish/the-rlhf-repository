import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import pygame

# import gymnasium as gym
import gym
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


def collect_human_demos(num_demos, demo_type, index):
    mapping = {(pygame.K_LEFT,): 0, (pygame.K_RIGHT,): 2}

    env = gym.make("CartPole-v1", render_mode="rgb_array")
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
    sas_pairs = ["{0}_video_{1}.mp4".format(demo_type, index)]
    video = VideoRecorder(
        env,
        "/Users/u1452582/Desktop/Human_AI_Alignment/the-rlhf-repository/improvement/improvement_data/{0}_video_{1}.mp4".format(
            demo_type, index
        ),
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
            obs, rew, done, info, _ = env.step(action)
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
    print("{0}_video_{1}.mp4 is done recording".format(demo_type, index))
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


if __name__ == "__main__":
    sas_pairs = []
    for i in range(5):
        sas_pairs.append(collect_human_demos(1, "improvement", i))

    print(sas_pairs)
    file = open(
        "/Users/u1452582/Desktop/Human_AI_Alignment/the-rlhf-repository/improvement/improvement_data/improvement_sas.txt",
        "w",
    )
    for x in sas_pairs:
        file.write(x + "\n")
    file.close()
