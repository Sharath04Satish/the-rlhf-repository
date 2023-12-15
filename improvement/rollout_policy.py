import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import gym
from vpg import mlp
import gymnasium as gym


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
            obs, rew, done, _, _ = env.step(act)
            cum_ret += rew
            obs_traj.append(obs)

    env.close()
    return obs_traj, cum_ret


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", "--env", type=str, default="CartPole-v1")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument(
        "--checkpoint", type=str, default="", help="pretrained policy weights"
    )
    parser.add_argument("--num_rollouts", type=int, default=1)

    args = parser.parse_args()

    checkpoint = args.checkpoint

    env = gym.make(args.env_name, render_mode="rgb_array")
    env.reset()
    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n
    hidden_sizes = [32]
    policy = mlp(sizes=[obs_dim] + hidden_sizes + [n_acts])
    device = "cpu"
    policy.load_state_dict(torch.load(checkpoint))

    returns = 0
    print("\n")
    for i in range(args.num_rollouts):
        _, cum_ret = generate_rollout(policy, env, i + 1, rendering=args.render)
        print("The cummulative returns are, ", cum_ret)
        returns += cum_ret
    print("\nThe average return is,", returns / args.num_rollouts)
