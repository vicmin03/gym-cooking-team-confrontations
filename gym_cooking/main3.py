from gym_cooking.envs import OvercookedEnvironment
from recipe_planner.recipe import *
from utils.world import World
from utils.agent2 import RealAgent, SimAgent, COLORS, TEAM_COLORS
from utils.core import *
from misc.game.gameplay import GamePlay
from misc.metrics.metrics_bag import Bag

import numpy as np
import random
import argparse
from collections import namedtuple, deque
import itertools

import gym
from dqn.network import Network
from dqn.replay_buffer import ReplayBuffer
from make_env import make_env
import gym
import torch as T
from torch import nn

def parse_arguments():
    parser = argparse.ArgumentParser("Overcooked 2 argument parser")

    # Environment
    parser.add_argument("--level", type=str, required=True)
    parser.add_argument("--num-agents", type=int, required=True)
    parser.add_argument("--max-num-timesteps", type=int, default=70, help="Max number of timesteps to run")
    parser.add_argument("--max-num-subtasks", type=int, default=14, help="Max number of subtasks for recipe")
    parser.add_argument("--seed", type=int, default=1, help="Fix pseudorandom seed")
    parser.add_argument("--with-image-obs", action="store_true", default=False, help="Return observations as images (instead of objects)")

    # Delegation Planner
    parser.add_argument("--beta", type=float, default=1.3, help="Beta for softmax in Bayesian delegation updates")

    # Navigation Planner
    parser.add_argument("--alpha", type=float, default=0.01, help="Alpha for BRTDP")
    parser.add_argument("--tau", type=int, default=2, help="Normalize v diff")
    parser.add_argument("--cap", type=int, default=75, help="Max number of steps in each main loop of BRTDP")
    parser.add_argument("--main-cap", type=int, default=100, help="Max number of main loops in each run of BRTDP")

    # Visualizations
    parser.add_argument("--play", action="store_true", default=False, help="Play interactive game with keys")
    parser.add_argument("--record", action="store_true", default=False, help="Save observation at each time step as an image in misc/game/record")

    # Models
    # Valid options: `bd` = Bayes Delegation; `up` = Uniform Priors
    # `dc` = Divide & Conquer; `fb` = Fixed Beliefs; `greedy` = Greedy, 'rl' = reinforcement learning
    parser.add_argument("--model2", type=str, default=None, help="Model type for agent 2 (bd, up, dc, fb, greedy or dqn)")
    parser.add_argument("--model3", type=str, default=None, help="Model type for agent 3 (bd, up, dc, fb, greedy or dqn)")
    parser.add_argument("--model4", type=str, default=None, help="Model type for agent 4 (bd, up, dc, fb, greedy or dqn)")
    parser.add_argument("--model1", type=str, default=None, help="Model type for agent 1 (bd, up, dc, fb, greedy or dqn)")
    parser.add_argument("--model5", type=str, default=None, help="Model type for agent 1 (bd, up, dc, fb, greedy or dqn)")
    parser.add_argument("--model6", type=str, default=None, help="Model type for agent 2 (bd, up, dc, fb, greedy or dqn)")
    parser.add_argument("--model7", type=str, default=None, help="Model type for agent 3 (bd, up, dc, fb, greedy or dqn)")
    parser.add_argument("--model8", type=str, default=None, help="Model type for agent 4 (bd, up, dc, fb, greedy or dqn)")

    # whether each team will have a hoarder agent (1st agent on each team)
    parser.add_argument("--hoarder", type=bool, default=False, help="Whether there is an agent performing hoarding for this team")

    return parser.parse_args()


def initialize_agents(arglist, env):
    real_agents = []

    with open('utils/levels/{}.txt'.format(arglist.level), 'r') as f:
        phase = 0
        recipes = []
        for line in f:
            line = line.strip('\n')
            if line == '':
                phase += 1

            elif phase == 0:
                if 'stock' not in line:
                    phase +=1

            # phase 2: read in recipe list
            elif phase == 2:
                recipes.append(globals()[line]())

            # Phase 3: Read whether teams (competitive) mode or coop mode
            elif phase == 3:

                if 'teams' in line:
                    phase = 4
                else:
                    phase = 5

            # phase 4: read in agent locations (up to num_agents) for teams
            elif phase == 4:
                if len(real_agents) < arglist.num_agents:
                    loc = line.split(' ')
                    if arglist.hoarder and len(real_agents) < 3:
                        hoarder = True
                    else:
                        hoarder = False
                    real_agent = RealAgent(
                            arglist=arglist,
                            name='agent-'+str(len(real_agents)+1),
                            id_color=TEAM_COLORS[len(real_agents) % 2][int(len(real_agents)/2)],
                            recipes=recipes, hoarder=hoarder, online_net=Network(env), target_net = Network(env))
                    real_agent.set_team((len(real_agents) % 2) + 1)
                    real_agents.append(real_agent)

            # phase 5: read in agent locations when not in teams
            elif phase == 5:
                if len(real_agents) < arglist.num_agents:
                    loc = line.split(' ')
                    real_agent = RealAgent(
                        arglist=arglist,
                        name='agent-' + str(len(real_agents) + 1),
                        id_color=COLORS[len(real_agents)],
                        recipes=recipes)
                    real_agents.append(real_agent)

    return real_agents


# --- Parameters ---------------------------------

GAMMA = 0.99     # the discount rate
BATCH_SIZE = 32    # no of samples we sample from the memory buffer
BUFFER_SIZE = 50000     # the max no. of samples stores in the buffer before overriding old transitions
# MIN_REPLAY_SIZE = 1000      # the no. of transitions we need in repay buffer before training can begin
MIN_REPLAY_SIZE = 33
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 10000   # epsilon decreases from start_val to end_val over this many steps
TARGET_UPDATE_FREQ = 1000   # how many steps before target network params are updated from online parameters

if __name__ == '__main__':
    arglist = parse_arguments()

    env = gym.envs.make("gym_cooking:overcookedEnv-v0", arglist=arglist)
    obs = env.reset()

    dqn_agents = initialize_agents(arglist=arglist, env=env)

    # replay buffer to keep track of action history and transitions
    replay_buffer = deque(maxlen=BUFFER_SIZE)
    
    # reward buffer keeps history of rewards, as tuples for rewards of each team (team1_reward, team2_reward)
    # reward_buffer = deque([0,0], maxlen=100)
    reward_buffer = deque(maxlen=100)

    team1_reward = 0
    team2_reward = 0

    # set up online and target networks
    # online_net = Network(env)
    # target_net = Network(env)

    # loads the parameters of the online network to the target network
    for agent in dqn_agents:
        agent.target_net.load_state_dict(agent.online_net.state_dict())


    # Initialise replay buffer by randomly choosing actions in the environment and saving rewards
    for i in range (MIN_REPLAY_SIZE):
        action_dict = {}
        action_arr = []
        for agent in dqn_agents:
            # choose a random action to perform in the environment 
            action = env.action_space.sample()
            action_dict[agent.name] = env.possible_actions[action]
            action_arr.append(action)

        # get observation from performing actions
        new_obs, reward1, reward2, done, info = env.step(action_dict, dqn_agents)

        # save transition as record in replay buffer
        # transition = (obs.create_obs(), action, reward1, reward2, done, new_obs.create_obs())
        transition = (obs.create_obs(), np.asarray(action_arr), reward1, reward2, done, new_obs.create_obs())
        print("Hey these are the actions chosen", np.asarray(action_arr))
        print("For agent 1:", action_arr[0], "For agent 2: ", action_arr[1])
        replay_buffer.append(transition)
        obs = new_obs

        if done:
            obs = env.reset()


    # training loop
    for step in itertools.count():
        # holds which action each agent takes
        action_dict = {}
        action_arr = []
        # this reduces epsilon based on what step we're on - decays from start to end value
        epsilon = np.interp(step, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])

        for agent in dqn_agents:
            random_sample = random.random()

            if random_sample <= epsilon:
                # take a random action = exploration
                action = env.action_space.sample()
            else:
                # or intelligently choose an action based on learning
                action = agent.online_net.select_action(obs)
            action_dict[agent.name] = env.possible_actions[action]
            action_arr.append(action)
            # get + save observation from performing action
                # done flag indicates whether a terminal state or not
        new_obs, reward1, reward2, done, info = env.step(action_dict, dqn_agents)

        
        # actionIndex = env.possible_actions.index((action[0], action[1]))
        # transition = (obs.create_obs(), action, reward1, reward2, done, new_obs.create_obs())
        transition = (obs.create_obs(), np.asarray(action_arr), reward1, reward2, done, new_obs.create_obs())
        replay_buffer.append(transition)
        obs = new_obs
        
        
        team1_reward += reward1
        team2_reward += reward2

        reward_buffer.append((team1_reward, team2_reward))  
        team1_reward = 0
        team2_reward = 0

        if done:
            obs = env.reset()


        # Start Gradient Step ----------
        # samples BATCH_SIZE number of transitions from the replay buffer into a list 
        transitions = random.sample(replay_buffer, BATCH_SIZE)

        # extract each part from the transition tuples into their own np array
        observations = np.asarray([t[0] for t in transitions])
        actions = np.asarray([t[1] for t in transitions])
        rewards1 = np.asarray([t[2] for t in transitions])
        rewards2 = np.asarray([t[3] for t in transitions])
        dones = np.asarray([t[4] for t in transitions])
        new_observations = np.asarray([t[5] for t in transitions])


        # turning into tensors
        observations_t = T.as_tensor(observations, dtype=T.float32) 
        actions_t = T.as_tensor(actions, dtype=T.int64).unsqueeze(-1)
        rewards1_t = T.as_tensor(rewards1, dtype=T.float32).unsqueeze(-1)
        rewards2_t = T.as_tensor(rewards2, dtype=T.float32).unsqueeze(-1)
        dones_t = T.as_tensor(dones, dtype=T.float32).unsqueeze(-1)
        new_observations_t = T.as_tensor(new_observations, dtype=T.float32)

        # Compute Targets
            # gets a set of q values for each observation, with q as the first dimension
        
        for i in range (0, len(dqn_agents)):
            agent = dqn_agents[i]
            target_q_values = agent.target_net(new_observations_t)
                # max returns a tuple with (highest_val, index)
            max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]


            # choose to use the rewards tensor of their own team only 
            if agent.team == 1:
                targets = rewards1_t * GAMMA * (1- dones_t) * max_target_q_values
            elif agent.team == 2:
                targets = rewards2_t * GAMMA * (1- dones_t) * max_target_q_values

            # Compute Loss
                # get expected q values from the online nn based on the given observations
            q_values = agent.online_net(observations_t)

            # get all the actions taken by this agent
            my_actions = [action[i] for action in actions]
            my_actions_t = T.as_tensor(my_actions, dtype=T.int64).unsqueeze(-1)

            # this applies the index of the actions taken by the agent (my_actions_t) to get the q_value for that action
            action_q_values = T.gather(input=q_values, dim=1, index=my_actions_t)

            # compute the loss with huber loss function
            loss = nn.functional.smooth_l1_loss(action_q_values, targets)

            # Gradient Descent
            agent.optimizer.zero_grad()
            loss.backward()
            agent.optimizer.step()

            # Update Target Network parameters
            if step % TARGET_UPDATE_FREQ == 0:
                # updates the target network params to be the same as the online network
                agent.target_net.load_state_dict(agent.online_net.state_dict())

        # Logging to check
        if step % 50 == 0:
            print()
            print("Step", step)
            sum1 = 0
            sum2 = 0
            for reward in reward_buffer:
                sum1 += reward[0]
                sum2 += reward[1]
                print("reward for team 1 is", sum1)
                print("reward for team 2 is", sum2)
                print("length of ", reward_buffer, "is, ", len(reward_buffer))
            # print("Average Reward for team 1: ", np.mean(reward[0] for reward in reward_buffer))
            print("Average Reward for team 1: ", sum1/len(reward_buffer))
            print("Average Reward for team 2: ", sum2/len(reward_buffer))

    
