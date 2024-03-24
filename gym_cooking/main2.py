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
from collections import namedtuple 

import gym
from cooking_maddpg.maddpg import MADDPG
from cooking_maddpg.experience_replay_buffer import ExperienceReplayBuffer
from make_env import make_env
import gym

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
    parser.add_argument("--model1", type=str, default=None, help="Model type for agent 1 (bd, up, dc, fb, or greedy)")
    parser.add_argument("--model2", type=str, default=None, help="Model type for agent 2 (bd, up, dc, fb, or greedy)")
    parser.add_argument("--model3", type=str, default=None, help="Model type for agent 3 (bd, up, dc, fb, or greedy)")
    parser.add_argument("--model4", type=str, default=None, help="Model type for agent 4 (bd, up, dc, fb, or greedy)")
    parser.add_argument("--model5", type=str, default=None, help="Model type for agent 1 (bd, up, dc, fb, or greedy)")
    parser.add_argument("--model6", type=str, default=None, help="Model type for agent 2 (bd, up, dc, fb, or greedy)")
    parser.add_argument("--model7", type=str, default=None, help="Model type for agent 3 (bd, up, dc, fb, or greedy)")
    parser.add_argument("--model8", type=str, default=None, help="Model type for agent 4 (bd, up, dc, fb, or greedy)")

    # whether each team will have a hoarder agent (1st agent on each team)
    parser.add_argument("--hoarder", type=bool, default=False, help="Whether there is an agent performing hoarding for this team")

    return parser.parse_args()



PRINT_INTERVAL = 500            
NUM_EPS = 50000     # maximum number of episodes to run
MAX_STEPS = 70   # maximum timesteps in a game
total_steps = 0
score1_history = [] 
score2_history = [] 
evaluate = False      # indicates whether just training or using to evaluate results
best_score = 0

# util function needed to combine observations of state, action
def obs_list_to_state_vector(observation):
    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs])
    return state

def train(env, maddpg_agents, buffer, NUM_EPS, MAX_STEPS, batch_size, gamma, tau):
    num_agents = len(maddpg_agents)

    for i in range(NUM_EPS):
        obs = env.reset()
        score = 0
        done = [False]*num_agents
        episode_step = 0
        while not any(done):

            actions = []
            for agent in maddpg_agents:
                actions.append(agent.select_action(obs.encode_obs(maddpg_agents)))
            obs_, reward1, reward2, done, info = env.step(actions)

            state = obs_list_to_state_vector(obs)
            state_ = obs_list_to_state_vector(obs_)

            if episode_step >= MAX_STEPS:
                done = [True]*num_agents

            buffer.store_transition(obs, state, actions, reward1, reward2, obs_, state_, done)

            if total_steps % 100 == 0 and not evaluate:
                maddpg_agents.learn(buffer)

            # obs2 = env.encode_obs(obs_, maddpg_agents)

            score1 += sum(reward1)
            score2 += sum(reward2)
            total_steps += 1
            episode_step += 1

        score1_history.append(score1)
        score2_history.append(score2)
        avg_score = np.mean(score1_history[-100:])
        if not evaluate:
            if avg_score > best_score:
                maddpg_agents.save_checkpoint()
                best_score = avg_score
        if i % PRINT_INTERVAL == 0 and i > 0:
            print('episode', i, 'average score {:.1f}'.format(avg_score))


def initialize_agents(arglist):
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
                            recipes=recipes, hoarder=hoarder)
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



if __name__ == '__main__':
    arglist = parse_arguments()

    env = gym.envs.make("gym_cooking:overcookedEnv-v0", arglist=arglist)
    obs = env.reset()

    MADDPG_agents = initialize_agents(arglist=arglist)


    # memory = ExperienceReplayBuffer(1000000, critic_dims, actor_dims, 
    #                     n_actions, n_agents, batch_size=1024)
    
    # actor_dims = env.
    
    memory = ExperienceReplayBuffer(buffer_size=1000000, num_agents=2, obs_dims=64, batch_size=1024)
    

    train(env, MADDPG_agents, memory, NUM_EPS, MAX_STEPS, batch_size=1024, gamma=0.95, tau=0.01)


    
