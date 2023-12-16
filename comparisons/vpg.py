import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gym
from gym.spaces import Discrete, Box
from utils_syn import mlp, Net
import os


def reward_to_go(rews):
    n = len(rews)
    rtgs = np.zeros_like(rews)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + (rtgs[i + 1] if i + 1 < n else 0)
    return rtgs


def train(
    env_name="CartPole-v1",
    hidden_sizes=[32],
    lr=1e-2,
    epochs=50,
    batch_size=5000,
    render=False,
    reward=None,
    checkpoint=False,
    checkpoint_dir="\.",
):
    env = gym.make(env_name, render_mode="rgb_array")
    assert isinstance(
        env.observation_space, Box
    ), "This example only works for envs with continuous state spaces."
    assert isinstance(
        env.action_space, Discrete
    ), "This example only works for envs with discrete action spaces."

    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    logits_net = mlp(sizes=[obs_dim] + hidden_sizes + [n_acts])

    def get_policy(obs):
        logits = logits_net(obs)
        return Categorical(logits=logits)

    def get_action(obs):
        return get_policy(obs).sample().item()

    def compute_loss(obs, act, weights):
        logp = get_policy(obs).log_prob(act)
        return -(logp * weights).mean()

    optimizer = Adam(logits_net.parameters(), lr=lr)

    def train_one_epoch():
        batch_obs = []
        batch_acts = []
        batch_weights = []
        batch_rets = []
        batch_lens = []

        obs = env.reset()
        if type(obs) is tuple:
            obs = obs[0]
        done = False
        ep_rews = []
        finished_rendering_this_epoch = False

        while True:
            if (not finished_rendering_this_epoch) and render:
                env.render()

            if type(obs) is tuple:
                obs = obs[0]

            batch_obs.append(obs.copy())
            act = get_action(torch.as_tensor(obs, dtype=torch.float32))
            obs, rew, done, _, _ = env.step(act)

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            torchified_state = torch.from_numpy(obs).float().to(device)

            if reward is not None:
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                torchified_state = torch.from_numpy(obs).float().to(device)
                r = reward.predict_reward(torchified_state.unsqueeze(0)).item()
                rew = r

            batch_acts.append(act)
            ep_rews.append(rew)

            if done:
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                batch_weights += list(reward_to_go(ep_rews))
                obs, done, ep_rews = env.reset(), False, []
                finished_rendering_this_epoch = True

                if len(batch_obs) > batch_size:
                    break

        optimizer.zero_grad()
        batch_loss = compute_loss(
            obs=torch.as_tensor(batch_obs, dtype=torch.float32),
            act=torch.as_tensor(batch_acts, dtype=torch.int32),
            weights=torch.as_tensor(batch_weights, dtype=torch.float32),
        )
        batch_loss.backward()
        optimizer.step()
        return batch_loss, batch_rets, batch_lens

    for i in range(epochs):
        batch_loss, batch_rets, batch_lens = train_one_epoch()
        if reward is not None:
            print(
                "Epoch: %3d \t Loss: %.3f \t Predicted Return: %.3f \t Episode Length (gt reward): %.3f"
                % (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens))
            )
        else:
            print(
                "Epoch: %3d \t Loss: %.3f \t Return: %.3f \t Episode Length: %.3f"
                % (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens))
            )

        if checkpoint:
            torch.save(
                logits_net.state_dict(),
                checkpoint_dir + "/policy_checkpoint" + str(i) + ".params",
            )

    if not checkpoint:
        torch.save(logits_net.state_dict(), checkpoint_dir + "/final_policy.params")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", "--env", type=str, default="CartPole-v1")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--checkpoint", action="store_true")
    parser.add_argument("--checkpoint_dir", type=str, default="\.")
    parser.add_argument(
        "--reward_params",
        type=str,
        default="",
        help="parameters of learned reward function",
    )
    args = parser.parse_args()

    isExist = os.path.exists(args.checkpoint_dir)
    if not isExist:
        os.makedirs(args.checkpoint_dir)

    if args.reward_params == "":
        train(
            env_name=args.env_name,
            render=args.render,
            lr=args.lr,
            epochs=args.epochs,
            checkpoint=args.checkpoint,
            checkpoint_dir=args.checkpoint_dir,
        )
    else:
        print("\nLet's train the agent based on the learned reward function.\n")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        reward_net = Net()
        reward_net.load_state_dict(torch.load(args.reward_params))
        reward_net.to(device)
        train(
            env_name=args.env_name,
            render=args.render,
            lr=args.lr,
            epochs=args.epochs,
            reward=reward_net,
            checkpoint=args.checkpoint,
            checkpoint_dir=args.checkpoint_dir,
        )
