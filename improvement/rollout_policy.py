import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gym
from gym.spaces import Discrete, Box
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from vpg import mlp
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


def get_cumulative_rewards_from_human_demonstrations(env, demos, rendering=False):
    obs = env.reset()

    cum_ret = 0
    obs_traj = []
    human_actions = [action for _, action, _ in demos]

    for action in human_actions:
        if rendering:
            env.render()

        obs, rew, _, _, _ = env.step(action)
        cum_ret += rew
        obs_traj.append(obs)

    return obs_traj, cum_ret


def generate_rollout_ppo_sgd(index):
    cum_ret = 0
    obs_traj = []
    dones = False

    vec_env = make_vec_env("CartPole-v1")

    video = VideoRecorder(
        vec_env,
        "../web_app/static/comparisons_data/{0}_demonstration_{1}.mp4".format(
            "synthetic", index
        ),
    )

    model = PPO("MlpPolicy", vec_env, verbose=1)
    model.learn(total_timesteps=25000)
    model.save("ppo_mountain_car")

    del model

    model = PPO.load("ppo_mountain_car")

    obs = vec_env.reset()
    while not dones:
        action, _states = model.predict(obs)
        video.capture_frame()
        obs, rewards, dones, info = vec_env.step(action)
        cum_ret += rewards
        obs_traj.append(obs)
        vec_env.render("human")

    video.close()

    return obs_traj, cum_ret


# execute the policy to generate one rollout and keep track of performance
def generate_rollout_1(policy, env, index, rendering=False):
    # make action selection function (outputs int action, sampled from policy)
    def get_action(policy, obs):
        logits = policy(obs)
        return Categorical(logits=logits).sample().item()

    # reset episode-specific variables
    obs = env.reset()  # first obs comes from starting distribution
    if type(obs) is tuple:
        obs = obs[0]
    done = False  # signal from environment that episode is over
    ep_rews = []  # list for rewards accrued throughout ep

    cum_ret = 0
    obs_traj = []

    video = VideoRecorder(
        env,
        "../web_app/static/comparisons_data/{0}_demonstration_{1}.mp4".format(
            "synthetic", index
        ),
    )

    # collect experience by acting in the environment with current policy
    while not done:
        # rendering
        if rendering:
            env.render()

        if len(obs) > 0:
            # act in the environment
            act = get_action(policy, torch.as_tensor(obs, dtype=torch.float32))
            # print(obs)
            video.capture_frame()
            obs, rew, done, _, _ = env.step(act)
            # print(rew, done, a, b)
            cum_ret += rew
            obs_traj.append(obs.tolist())

    video.close()
    env.close()

    return obs_traj, cum_ret


# execute the policy to generate one rollout and keep track of performance
def generate_rollout(policy, env, index, rendering=False):
    # make action selection function (outputs int action, sampled from policy)
    def get_action(policy, obs):
        logits = policy(obs)
        return Categorical(logits=logits).sample().item()

    # reset episode-specific variables
    obs = env.reset()  # first obs comes from starting distribution
    if type(obs) is tuple:
        obs = obs[0]
    done = False  # signal from environment that episode is over
    ep_rews = []  # list for rewards accrued throughout ep

    cum_ret = 0
    obs_traj = []

    # video = VideoRecorder(
    #     env, "comparisons_data/{0}_demonstration_{1}.mp4".format("synthetic", index)
    # )

    # collect experience by acting in the environment with current policy
    while not done:
        # rendering
        if rendering:
            env.render()

        if len(obs) > 0:
            # act in the environment
            act = get_action(policy, torch.as_tensor(obs, dtype=torch.float32))
            # print(obs)
            # video.capture_frame()
            obs, rew, done, a, _ = env.step(act)
            # print(rew, done, a, b)
            cum_ret += rew
            obs_traj.append(obs)

    # video.close()
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

    # make core of policy network
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
        print("cumulative return", cum_ret)
        returns += cum_ret
    print("average return", returns / args.num_rollouts)
