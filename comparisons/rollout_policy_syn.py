import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from vpg import mlp
import gymnasium as gym


def generate_rollout_1(policy, env, index, rendering=False):
    def get_action(policy, obs):
        logits = policy(obs)
        return Categorical(logits=logits).sample().item()

    obs = env.reset()
    if type(obs) is tuple:
        obs = obs[0]
    done = False
    cum_ret = 0
    obs_traj = []

    video = VideoRecorder(
        env,
        "../web_app/static/comparisons_data/{0}_demonstration_{1}.mp4".format(
            "synthetic", index
        ),
    )

    while not done:
        if rendering:
            env.render()

        if len(obs) > 0:
            act = get_action(policy, torch.as_tensor(obs, dtype=torch.float32))
            video.capture_frame()
            obs, rew, done, _, _ = env.step(act)
            cum_ret += rew
            obs_traj.append(obs.tolist())

    video.close()
    env.close()

    return obs_traj, cum_ret


def generate_rollout(policy, env, index, rendering=False):
    def get_action(policy, obs):
        logits = policy(obs)
        return Categorical(logits=logits).sample().item()

    obs = env.reset()
    if type(obs) is tuple:
        obs = obs[0]
    done = False

    cum_ret = 0
    obs_traj = []

    while not done:
        if rendering:
            env.render()

        if len(obs) > 0:
            act = get_action(policy, torch.as_tensor(obs, dtype=torch.float32))
            obs, rew, done, a, _ = env.step(act)
            cum_ret += rew
            obs_traj.append(obs)

    env.close()

    return obs_traj, cum_ret


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", "--env", type=str, default="CartPole-v1")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument(
        "--checkpoint", type=str, default="", help="pretrained policy weights"
    )
    parser.add_argument("--num_rollouts", type=int, default=1)

    args = parser.parse_args()

    checkpoint = args.checkpoint

    env = gym.make(args.env_name)
    env.reset()
    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n
    hidden_sizes = [32]
    policy = mlp(sizes=[obs_dim] + hidden_sizes + [n_acts])
    device = "cpu"
    policy.load_state_dict(torch.load(checkpoint))

    returns = 0
    for i in range(args.num_rollouts):
        _, cum_ret = generate_rollout(policy, env, i + 1, rendering=args.render)
        print("\nThe cumulative return is,", cum_ret)
        returns += cum_ret
    print("\nThe average return is,", returns / args.num_rollouts)
