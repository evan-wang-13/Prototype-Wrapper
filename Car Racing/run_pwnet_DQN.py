import gym
import torch
import torch.nn as nn
import numpy as np
import pickle
import toml

from copy import deepcopy
from torch.utils.data import TensorDataset, DataLoader
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


NUM_ITERATIONS = 1
MODEL_DIR = "weights/pw_net_DQN.pth"
CONFIG_FILE = "config.toml"
NUM_CLASSES = 3
LATENT_SIZE = 216
PROTOTYPE_SIZE = 50
BATCH_SIZE = 16
NUM_EPOCHS = 50
DEVICE = "cpu"
delay_ms = 0
NUM_PROTOTYPES = 4
SIMULATION_EPOCHS = 30


class PWNet(nn.Module):
    def __init__(self):
        super(PWNet, self).__init__()
        self.ts = ListModule(self, "ts_")
        for i in range(NUM_PROTOTYPES):
            transformation = nn.Sequential(
                nn.Linear(LATENT_SIZE, PROTOTYPE_SIZE),
                nn.InstanceNorm1d(PROTOTYPE_SIZE),
                nn.ReLU(),
                nn.Linear(PROTOTYPE_SIZE, PROTOTYPE_SIZE),
            )
            self.ts.append(transformation)
        self.epsilon = 1e-5
        self.linear = nn.Linear(NUM_PROTOTYPES, NUM_CLASSES, bias=False)
        self.__make_linear_weights()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.nn_human_x = nn.Parameter(
            torch.randn(NUM_PROTOTYPES, LATENT_SIZE), requires_grad=False
        )

    def __make_linear_weights(self):
        """
        Must be manually defined to connect prototypes to human-friendly concepts
        For example, -1 here corresponds to steering left, whilst the 1 below it to steering right
        Together, they can encapsulate the overall concept of steering
        More could be connected, but we just use 2 here for simplicity.
        """

        custom_weight_matrix = torch.tensor(
            [
                [-1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        self.linear.weight.data.copy_(custom_weight_matrix.T)

    def __proto_layer_l2(self, x, p):
        output = list()
        b_size = x.shape[0]
        p = p.view(1, PROTOTYPE_SIZE).tile(b_size, 1).to(DEVICE)
        c = x.view(b_size, PROTOTYPE_SIZE).to(DEVICE)
        l2s = ((c - p) ** 2).sum(axis=1).to(DEVICE)
        act = torch.log((l2s + 1.0) / (l2s + self.epsilon)).to(DEVICE)
        return act

    def __output_act_func(self, p_acts):
        """
        Use appropriate activation functions for the problem at hand
        Here, tanh and relu make the most sense as they bin the possible output
        ranges to be what the car is capable of doing.
        """

        p_acts.T[0] = self.tanh(p_acts.T[0])  # steering between -1 -> +1
        p_acts.T[1] = self.relu(p_acts.T[1])  # acc > 0
        p_acts.T[2] = self.relu(p_acts.T[2])  # brake > 0
        return p_acts

    def forward(self, x):
        # Get the latent prototypes by putting them through the individual transformations
        trans_nn_human_x = list()
        for i, t in enumerate(self.ts):
            trans_nn_human_x.append(
                t(torch.tensor(self.nn_human_x[i], dtype=torch.float32).view(1, -1))
            )
        latent_protos = torch.cat(trans_nn_human_x, dim=0)

        # Do similarity of inputs to prototypes
        p_acts = list()
        for i, t in enumerate(self.ts):
            action_prototype = latent_protos[i]
            p_acts.append(self.__proto_layer_l2(t(x), action_prototype).view(-1, 1))
        p_acts = torch.cat(p_acts, axis=1)

        # Put though activation function method
        logits = self.linear(p_acts)
        final_outputs = self.__output_act_func(logits)

        return final_outputs


def evaluate_loader(model, loader, loss):
    model.eval()
    total_error = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(loader):
            imgs, labels = data
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            logits = model(imgs)
            current_loss = loss(logits, labels)
            total_error += current_loss.item()
            total += len(imgs)
    model.train()
    return total_error / total


def load_config():
    with open(CONFIG_FILE, "r") as f:
        config = toml.load(f)
    return config


class ListModule(object):
    # Should work with all kind of module
    def __init__(self, module, prefix, *args):
        self.module = module
        self.prefix = prefix
        self.num_module = 0
        for new_module in args:
            self.append(new_module)

    def append(self, new_module):
        if not isinstance(new_module, nn.Module):
            raise ValueError("Not a Module")
        else:
            self.module.add_module(self.prefix + str(self.num_module), new_module)
            self.num_module += 1

    def __len__(self):
        return self.num_module

    def __getitem__(self, i):
        if i < 0 or i >= self.num_module:
            raise IndexError("Out of bound")
        return getattr(self.module, self.prefix + str(i))


def proto_loss(model, nn_human_x, criterion):
    model.eval()
    target_x = trans_human_concepts(model, nn_human_x)
    loss = criterion(model.prototypes, target_x)
    model.train()
    return loss


def trans_human_concepts(model, nn_human_x):
    model.eval()
    trans_nn_human_x = list()
    for i, t in enumerate(model.ts):
        trans_nn_human_x.append(
            t(torch.tensor(nn_human_x[i], dtype=torch.float32).view(1, -1))
        )
    model.train()
    return torch.cat(trans_nn_human_x, dim=0)


#### Start Collecting Data To Form Final Mean and Standard Error Results

data_rewards = list()
data_errors = list()


def find_closest_action(batch_continuous_output, action_space):
    # Convert the action space to a tensor if it's not already
    action_space_tensor = torch.tensor(
        action_space, dtype=torch.float32, device=batch_continuous_output.device
    )

    # Ensure batch_continuous_output is a tensor (it should already be)
    batch_continuous_output_tensor = batch_continuous_output

    # Expand dimensions to allow broadcasting for distance calculation
    expanded_actions = action_space_tensor.unsqueeze(
        0
    )  # Shape: [1, num_actions, num_features]
    expanded_outputs = batch_continuous_output_tensor.unsqueeze(
        1
    )  # Shape: [batch_size, 1, num_features]

    # Compute the Euclidean distances between each pair
    distances = torch.norm(
        expanded_actions - expanded_outputs, dim=2
    )  # Shape: [batch_size, num_actions]

    # Find the indices of the actions with the minimum distance for each batch item
    closest_action_indices = torch.argmin(distances, axis=1)  # Shape: [batch_size]

    # Optional: To return actual action vectors instead of indices, uncomment the following line:
    return action_space_tensor[closest_action_indices]


# Define your action space
action_space = [
    (-1, 1, 0.2),
    (0, 1, 0.2),
    (1, 1, 0.2),
    (-1, 1, 0),
    (0, 1, 0),
    (1, 1, 0),
    (-1, 0, 0.2),
    (0, 0, 0.2),
    (1, 0, 0.2),
    (-1, 0, 0),
    (0, 0, 0),
    (1, 0, 0),
]


for _ in range(NUM_ITERATIONS):
    cfg = load_config()
    # env = CarRacing(
    #     frame_skip=0,
    #     frame_stack=4,
    # )
    env = gym.make("CarRacing-v1")
    agent = CarRacingDQNAgent(
        epsilon=0
    )  # Set epsilon to 0 to ensure all actions are instructed by the agent
    model = "weights/trial_500.h5"
    agent.load(model)

    with open("data/X_train.pkl", "rb") as f:
        X_train = pickle.load(f)
    with open("data/real_actions.pkl", "rb") as f:
        real_actions = pickle.load(f)

    X_train = np.array([item for sublist in X_train for item in sublist])
    real_actions = np.array([item for sublist in real_actions for item in sublist])
    tensor_x = torch.Tensor(X_train)
    tensor_y = torch.tensor(real_actions, dtype=torch.float32)
    train_dataset = TensorDataset(tensor_x, tensor_y)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)

    # Human defined Prototypes for interpretable model (these were gotten manually earlier)
    # A clustering analysis could be done to help guide the search, or they can be manually chosen.
    # Lastly, they could also be learned as pwnet* shows in the comparitive tests
    p_idxs = np.array([10582, 20116, 4616, 2659])
    nn_human_x = X_train[p_idxs.flatten()]
    nn_human_actions = real_actions[p_idxs.flatten()]

    #### Training
    model = PWNet().eval()
    model.nn_human_x.data.copy_(torch.tensor(nn_human_x))

    mse_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01,
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)
    best_error = float("inf")
    model.train()

    # Freeze Linear Layer
    model.linear.weight.requires_grad = False

    for epoch in range(NUM_EPOCHS):
        running_loss = 0
        model.eval()
        train_error = evaluate_loader(model, train_loader, mse_loss)
        model.train()

        if train_error < best_error:
            torch.save(model.state_dict(), "weights/pw_net_DQN.pth")
            best_error = train_error

        for instances, labels in train_loader:
            optimizer.zero_grad()

            instances, labels = instances.to(DEVICE), labels.to(DEVICE)

            logits = model(instances)

            # discretized_action = find_closest_action(logits, action_space)
            # print(discretized_action.shape)
            # print(labels.shape)
            loss = mse_loss(logits, labels)
            # loss = mse_loss(discretized_action, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print("Epoch:", epoch)
        print("Running Error:", running_loss / len(train_loader))
        print("MAE:", train_error)
        print(" ")
        scheduler.step()

    states, actions, rewards, log_probs, values, dones, X_train = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    # Wapper model with learned weights
    model = PWNet().eval()
    model.load_state_dict(torch.load(MODEL_DIR))
    print("Sanity Check: MSE Eval:", evaluate_loader(model, train_loader, mse_loss))

    reward_arr = []
    all_errors = list()

    for i in tqdm(range(SIMULATION_EPOCHS)):
        next_state = env.reset()
        next_state = process_state_image(next_state)
        count = 0
        rew = 0
        model.eval()

        state_frame_stack_queue = deque(
            [next_state] * agent.frame_stack_num, maxlen=agent.frame_stack_num
        )

        for t in range(10000):
            # Get black box action
            current_state_frame_stack = generate_state_frame_stack_from_queue(
                state_frame_stack_queue
            )
            count += 1

            bb_action, latent = agent.act(current_state_frame_stack)

            action = model(torch.tensor(latent))

            discretized_action = find_closest_action(action, action_space)

            all_errors.append(
                mse_loss(torch.tensor(bb_action), discretized_action).detach().item()
            )

            state, reward, done, info = env.step(discretized_action)

            # all_errors.append(
            #     mse_loss(torch.tensor(bb_action), action[0]).detach().item()
            # )

            # state, reward, done, info = env.step(action[0].detach().numpy())
            # state = ppo._to_tensor(state)
            rew += reward
            count += 1
            if done:
                break

        reward_arr.append(rew)

    data_rewards.append(sum(reward_arr) / SIMULATION_EPOCHS)
    data_errors.append(sum(all_errors) / SIMULATION_EPOCHS)
    print("Iteration Reward:", sum(reward_arr) / SIMULATION_EPOCHS)

data_errors = np.array(data_errors)
data_rewards = np.array(data_rewards)

print(" ")
print("===== Data MAE:")
print("Mean:", data_errors.mean())
print("Standard Error:", data_errors.std() / np.sqrt(NUM_ITERATIONS))
print(" ")
print("===== Data Reward:")
print("Rewards:", data_rewards)
print("Mean:", data_rewards.mean())
print("Standard Error:", data_rewards.std() / np.sqrt(NUM_ITERATIONS))
