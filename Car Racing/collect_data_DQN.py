import toml
import numpy as np
import torch
import pickle
import os
import gym

from argparse import ArgumentParser
from os.path import join
from games.carracing import RacingNet, CarRacing
from games.carracing_DQN import DQNNetwork, CarRacingDQNAgent
from ppo import PPO
from torch.distributions import Beta
from tqdm import tqdm
from tensorflow.keras.models import load_model
import torch.nn as nn
from common_functions import process_state_image, generate_state_frame_stack_from_queue
from collections import deque


CONFIG_FILE = "config.toml"
device = "cpu"
NUM_EPISODES = 30

# if not os.path.exists('weights/'):
#     os.mkdir('weights/')
# if not os.path.exists('data/'):
#     os.mkdir('data/')


def load_config():
    with open(CONFIG_FILE, "r") as f:
        config = toml.load(f)
    return config


# def set_pytorch_weights(pytorch_model, tf_weights):
#     conv_and_fc_layers = [
#         module
#         for module in pytorch_model.modules()
#         if isinstance(module, (nn.Conv2d, nn.Linear))
#     ]

#     for layer, weights in zip(conv_and_fc_layers, tf_weights):
#         if isinstance(layer, nn.Conv2d):
#             # Transpose the weights from TF format (H, W, in_channels, out_channels) to PyTorch format (out_channels, in_channels, H, W)
#             transposed_weights = np.transpose(weights[0], (3, 2, 0, 1))
#             layer.weight.data = torch.from_numpy(transposed_weights).float()
#             if layer.bias is not None:
#                 layer.bias.data = torch.from_numpy(weights[1]).float()
#         elif isinstance(layer, nn.Linear):
#             # Transpose the weights for fully connected layers from TF (input_features, units) to PyTorch (units, input_features)
#             transposed_weights = np.transpose(weights[0])
#             layer.weight.data = torch.from_numpy(transposed_weights).float()
#             layer.bias.data = torch.from_numpy(weights[1]).float()


# cfg = load_config()
# env = CarRacing(frame_skip=0, frame_stack=3)
# net = DQNNetwork(3, env.action_space.shape)
# print("Action space: ", env.action_space.shape)

# # Extract weights
# tf_weights = []
# for layer in tf_model.layers:
#     weights = layer.get_weights()
#     if weights:  # Only conv and dense layers have weights
#         tf_weights.append(weights)


# Load the TensorFlow model
agent = CarRacingDQNAgent(
    epsilon=0
)  # Set epsilon to 0 to ensure all actions are instructed by the agent
model = "weights/trial_500.h5"
agent.load(model)
tf_model = agent.model


# ppo.load("weights/final_weights.pt")


# set_pytorch_weights(net, tf_weights)


env = gym.make("CarRacing-v1")
agent = CarRacingDQNAgent(
    epsilon=0
)  # Set epsilon to 0 to ensure all actions are instructed by the agent
agent.load(model)

real_actions, X_train, real_actions_index = [], [], []
# self_state = torch.tensor(env.reset(), dtype=torch.float32, device=device).unsqueeze(0)
reward_arr = list()

# print(self_state.shape)
# print(np.expand_dims(self_state, axis=0).shape)

# print(net(torch.tensor(env.reset(), dtype=torch.float32, device=device).unsqueeze(0)).shape)
for ep in tqdm(range(NUM_EPISODES)):
    next_state = env.reset()
    next_state = process_state_image(next_state)

    rew = 0
    done = False
    count = 0

    ep_actions = list()
    ep_actions_index = list()
    ep_x = list()
    state_frame_stack_queue = deque(
        [next_state] * agent.frame_stack_num, maxlen=agent.frame_stack_num
    )

    while not done:
        # env.render()
        current_state_frame_stack = generate_state_frame_stack_from_queue(
            state_frame_stack_queue
        )
        count += 1

        action, latent, action_index = agent.act(current_state_frame_stack)

        # Run one step of the environment based on the current policy
        # x is the final embedding produced by the network before converting to action dimensions. So x is the latent representation that the network encoder produces, = z in the paper
        # vector of size 256
        # value, alpha, beta, x = net(self_state)
        # value, alpha, beta = value.squeeze(0), alpha.squeeze(0), beta.squeeze(0)
        # policy = Beta(alpha, beta)

        # # Choose how to get actions (sample or take mean)
        # # input_action = policy.mean.detach()
        next_state, reward, done, info = env.step(action)

        rew += reward

        next_state = process_state_image(next_state)
        state_frame_stack_queue.append(next_state)

        # print("Input action shape: ", input_action.shape)

        # # Store the transition
        # print("ACTION: ", action)
        ep_actions.append(list(action))
        ep_actions_index.append(action_index)
        ep_x.append(latent.tolist()[0])
        # ep_states.append(self_state)

        # self_state = next_state
        # rew += reward

    print("Reward for episode " + str(ep) + ": ", str(rew))
    reward_arr.append(rew)
    # print(count)

    # Store the transition
    # states.append(ep_states)
    real_actions.append(ep_actions)
    X_train.append(ep_x)
    real_actions_index.append(ep_actions_index)
    # rew += reward


print("average reward per episode :", sum(reward_arr) / NUM_EPISODES)

# X_train: contains the latent states. List, one list for each episode
with open("data/X_train_DQN.pkl", "wb") as f:
    pickle.dump(X_train, f)
with open("data/real_actions_DQN.pkl", "wb") as f:
    pickle.dump(real_actions, f)  # 30 x 3 list
with open("data/real_actions_index_DQN.pkl", "wb") as f:
    pickle.dump(real_actions_index, f)
# with open('data/obs_train.pkl', 'wb') as f:
# 	pickle.dump(states, f)
# with open('data/saved_materials.pkl', 'wb') as f:
# 	pickle.dump(saved_materials, f)
