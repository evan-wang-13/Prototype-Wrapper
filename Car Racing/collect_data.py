import toml
import numpy as np
import torch
import pickle
import os

from argparse import ArgumentParser
from os.path import join
from games.carracing import RacingNet, CarRacing
from ppo import PPO
from torch.distributions import Beta
from tqdm import tqdm


CONFIG_FILE = "config.toml"
device = "cpu"
NUM_EPISODES = 30

if not os.path.exists("weights/"):
    os.mkdir("weights/")
if not os.path.exists("data/"):
    os.mkdir("data/")


def load_config():
    with open(CONFIG_FILE, "r") as f:
        config = toml.load(f)
    return config


cfg = load_config()
env = CarRacing(frame_skip=0, frame_stack=4)
net = RacingNet(
    env.observation_space.shape, env.action_space.shape
)  # action space shape is (2,) due to their modified env CarRacing
ppo = PPO(
    env,
    net,
    lr=cfg["lr"],
    gamma=cfg["gamma"],
    batch_size=cfg["batch_size"],
    gae_lambda=cfg["gae_lambda"],
    clip=cfg["clip"],
    value_coef=cfg["value_coef"],
    entropy_coef=cfg["entropy_coef"],
    epochs_per_step=cfg["epochs_per_step"],
    num_steps=cfg["num_steps"],
    horizon=cfg["horizon"],
    save_dir=cfg["save_dir"],
    save_interval=cfg["save_interval"],
)

ppo.load("weights/final_weights.pt")
states, real_actions, rewards, X_train = [], [], [], []
self_state = ppo._to_tensor(env.reset())
reward_arr = list()


for ep in tqdm(range(NUM_EPISODES)):
    next_state = ppo.env.reset()
    rew = 0
    done = False
    count = 0

    ep_actions = list()
    ep_x = list()

    while not done:
        count += 1

        # Run one step of the environment based on the current policy
        # x is the final embedding produced by the network before converting to action dimensions. So x is the latent representation that the network encoder produces, = z in the paper
        # vector of size 256
        value, alpha, beta, x = ppo.net(self_state)
        value, alpha, beta = value.squeeze(0), alpha.squeeze(0), beta.squeeze(0)
        policy = Beta(alpha, beta)

        # Choose how to get actions (sample or take mean)
        input_action = policy.mean.detach()
        # input_action = policy.sample()

        # print("Input Action Shape: ", input_action)
        next_state, reward, done, info, real_action = ppo.env.step(
            input_action.cpu().numpy()
        )
        next_state = ppo._to_tensor(next_state)

        # # Store the transition
        ep_actions.append(real_action.tolist())
        ep_x.append(x.tolist()[0])
        # ep_states.append(self_state)

        self_state = next_state
        rew += reward

        # ppo.env.render()

    reward_arr.append(rew)
    print(count)

    # Store the transition
    # states.append(ep_states)
    real_actions.append(ep_actions)
    X_train.append(ep_x)
    # rew += reward


print("average reward per episode :", sum(reward_arr) / NUM_EPISODES)

# X_train: contains the latent states. List, one list for each episode
with open("data/X_train_PPO.pkl", "wb") as f:
    pickle.dump(X_train, f)  # list, 30 x (num states) x 256
with open("data/real_actions_PPO.pkl", "wb") as f:
    pickle.dump(real_actions, f)  # list, 30 x (num states) x 3
# with open('data/obs_train.pkl', 'wb') as f:
# 	pickle.dump(states, f)
# with open('data/saved_materials.pkl', 'wb') as f:
# 	pickle.dump(saved_materials, f)
