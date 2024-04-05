from torch import nn
import torch as T
import gym
from collections import deque
import itertools
import numpy as np
import random


class Network(nn.Module):
    def __init__(self, env, path):
        super().__init__()
        
        # number of inputs to nn = product of features in observation space + number of agents (subtasks)
        in_features = 1
        for i in env.observation_space.shape:
            in_features *= i
        # in_features += len(env.get_agent_names())
        in_features += 1    # add the subtask of the relevant agent
        
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),            # input layer
            nn.Tanh(),
            # no. of outputs = the number of possible actions in the env space
            nn.Linear(64, env.action_space.n)
            # nn.Linear(64, 5)
        )
        
        # path where parameters are saved to and loaded from
        self.path = path


    def forward(self, x):
        return self.net(x)
    
    # determining what action to take
    def select_action(self, obs_v):
        # turn observation into a tensor (vector of each dimension value)
        obs_t = T.as_tensor(obs_v, dtype=T.float32)

        # calculate the q values for actions from this observation 
            # need unsequeeze to get batch dimension?
        q_values = self(obs_t.unsqueeze(0))

        max_q_index = T.argmax(q_values, dim=1)[0]
        action = max_q_index.detach().item()
        
        # return the index of the action with highest q values
        return action
    
    def save_params(self, map):
        print('./'+map+self.path)
        T.save(self.state_dict(), './'+map+'-'+self.path)

    def load_params(self, map):
        print("Tried to load", './'+map+'-'+self.path)
        self.load_state_dict(T.load('./'+map+'-'+self.path))