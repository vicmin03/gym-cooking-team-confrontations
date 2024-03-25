from torch import nn
import torch
import gym
from collections import deque
import itertools
import numpy as np
import random


class Network(nn.Module):
    def __init__(self, env):
        super().__init__()
        
        # number of inputs to nn = product of features in observation space
        in_features = 0
        for i in env.observation_space:
            for x in i.shape:
                in_features += x
        print(in_features, " = number of input features")
        # in_features = int(np.prod(sum(x for x in features)))
        # in_features = 
        
        self.net = nn.Sequential(
            # input layer
            nn.Linear(in_features, 64),
            nn.Tanh(),
            # no. of outputs = the number of possible actions in the env space
            # nn.Linear(64, env.action_space.n)
            nn.Linear(64, 5)
        )


    def forward(self, x):
        return self.net(x)
    
    # determining what action to take
    def select_action(self, obs):
        # turn observation into a tensor (vector of each dimension value)
        obs_t = torch.as_tensor(obs, dtype=torch.float32)

        # calculate the q values for actions from this observation 
            # need unsequeeze to get batch dimension?
        q_values = self(obs_t.unsqueeze(0))

        max_q_index = torch.argmax(q_values, dim=1)[0]
        action = max_q_index.detach().item()
        
        # return the index of the action with highest q values
        return action