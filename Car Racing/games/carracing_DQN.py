import torch
from torch import nn
import gym
from gym.spaces import Box
import numpy as np
from collections import deque
from copy import deepcopy

import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model


class DQNNetwork(nn.Module):
    def __init__(self, frame_stack_num, action_size):
        super().__init__()
        self.conv1 = nn.Conv2d(frame_stack_num, 6, kernel_size=7, stride=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 12, kernel_size=4)
        self.fc1 = nn.Linear(
            12 * 9 * 9, 216
        )  # Adjust the size according to output from conv layers
        self.fc2 = nn.Linear(216, action_size[0])

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        action = self.fc2(x)
        return action, x


class CarRacingDQNAgent:
    def __init__(
        self,
        action_space=[
            (-1, 1, 0.2),
            (0, 1, 0.2),
            (1, 1, 0.2),  #           Action Space Structure
            (-1, 1, 0),
            (0, 1, 0),
            (1, 1, 0),  #        (Steering Wheel, Gas, Break)
            (-1, 0, 0.2),
            (0, 0, 0.2),
            (1, 0, 0.2),  # Range        -1~1       0~1   0~1
            (-1, 0, 0),
            (0, 0, 0),
            (1, 0, 0),
        ],
        frame_stack_num=3,
        memory_size=5000,
        gamma=0.95,  # discount rate
        epsilon=1.0,  # exploration rate
        epsilon_min=0.1,
        epsilon_decay=0.9999,
        learning_rate=0.001,
    ):
        self.action_space = action_space
        self.frame_stack_num = frame_stack_num
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

    def build_model(self):
        input_layer = Input(shape=(96, 96, self.frame_stack_num))
        x = Conv2D(filters=6, kernel_size=(7, 7), strides=3, activation="relu")(
            input_layer
        )
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(filters=12, kernel_size=(4, 4), activation="relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Flatten()(x)
        latent_vector = Dense(216, activation="relu")(x)
        action_output = Dense(len(self.action_space), activation=None)(latent_vector)

        model = Model(inputs=input_layer, outputs=[action_output, latent_vector])
        model.compile(
            loss="mean_squared_error",
            optimizer=Adam(lr=self.learning_rate, epsilon=1e-7),
        )
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append(
            (state, self.action_space.index(action), reward, next_state, done)
        )

    def act(self, state):
        # if np.random.rand() > self.epsilon:
        act_values, latent_vector = self.model.predict(
            np.expand_dims(state, axis=0), verbose=0
        )
        action_index = np.argmax(act_values[0])
        # else:
        #     action_index = random.randrange(len(self.action_space))
        return self.action_space[action_index], latent_vector, action_index

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        train_state = []
        train_target = []
        for state, action_index, reward, next_state, done in minibatch:
            target = self.model.predict(np.expand_dims(state, axis=0))[0]
            if done:
                target[action_index] = reward
            else:
                t = self.target_model.predict(np.expand_dims(next_state, axis=0))[0]
                target[action_index] = reward + self.gamma * np.amax(t)
            train_state.append(state)
            train_target.append(target)
        self.model.fit(
            np.array(train_state), np.array(train_target), epochs=1, verbose=0
        )
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)
        self.update_target_model()

    def save(self, name):
        self.target_model.save_weights(name)
