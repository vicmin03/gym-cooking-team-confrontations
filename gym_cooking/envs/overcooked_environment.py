# Recipe planning
from recipe_planner.stripsworld import STRIPSWorld
import recipe_planner.utils as recipe
from recipe_planner.recipe import *

# Delegation planning
from delegation_planner.bayesian_delegator import BayesianDelegator

# Navigation planning
import navigation_planner.utils as nav_utils

# Other core modules
from utils.interact import interact
from utils.world import World
from utils.core import *
from utils.agent import SimAgent
from misc.game.gameimage import GameImage
from utils.agent import COLORS, TEAM_COLORS
from utils.utils import rep_to_int

import copy
import networkx as nx
import numpy as np
from itertools import combinations, permutations, product
from collections import namedtuple

import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym.spaces import Box, Discrete, Dict, Tuple, MultiBinary, MultiDiscrete
import torch as T


CollisionRepr = namedtuple("CollisionRepr", "time agent_names agent_locations")


class OvercookedEnvironment(gym.Env):
    """Environment object for Overcooked."""

    def __init__(self, arglist):
        self.arglist = arglist
        self.t = 0
        self.set_filename()

        # For visualizing episode.
        self.rep = []

        # For tracking data during an episode.
        self.collisions = []
        self.termination_info = ""
        self.successful = False

        # scores for each team
        self.team1_score = 0
        self.team2_score = 0

        # names of agents on each team
        self.team1_agents = []
        self.team2_agents = []

        # keep track of the dishes delivered for each team 
        self.delivered1 = 0
        self.delivered2 = 0

        # all possible actions in this environment
        self.action_space = Discrete(5)
        self.possible_actions = [(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)]


    def get_repr(self):
        return self.world.get_repr() + tuple([agent.get_repr() for agent in self.sim_agents])

    def __str__(self):
        # Print the world and agents.
        _display = list(map(lambda x: ''.join(map(lambda y: y + ' ', x)), self.rep))
        return '\n'.join(_display)

    def __eq__(self, other):
        return self.get_repr() == other.get_repr()

    def __copy__(self):
        new_env = OvercookedEnvironment(self.arglist)
        new_env.__dict__ = self.__dict__.copy()
        new_env.world = copy.copy(self.world)
        new_env.sim_agents = [copy.copy(a) for a in self.sim_agents]
        new_env.distances = self.distances

        # Make sure new objects and new agents' holdings have the right pointers.
        for a in new_env.sim_agents:
            if a.holding is not None:
                a.holding = new_env.world.get_object_at(
                        location=a.location,
                        desired_obj=None,
                        find_held_objects=True)
        return new_env

    def set_filename(self):
        self.filename = "{}_agents{}_seed{}".format(self.arglist.level,\
            self.arglist.num_agents, self.arglist.seed)
        model = ""
        if self.arglist.model1 is not None:
            model += "_model1-{}".format(self.arglist.model1)
        if self.arglist.model2 is not None:
            model += "_model2-{}".format(self.arglist.model2)
        if self.arglist.model3 is not None:
            model += "_model3-{}".format(self.arglist.model3)
        if self.arglist.model4 is not None:
            model += "_model4-{}".format(self.arglist.model4)
        self.filename += model

    def load_level(self, level, num_agents):
        x = 0
        y = 0
        with open('utils/levels/{}.txt'.format(level), 'r') as file:
            # Mark the phases of reading.
            phase = 0

            # controls stock of objects in map
            stock = 1

            for line in file:
                line = line.strip('\n')
                
                if line == '':
                    phase += 1

                elif phase == 0:
                    if 'stock' in line:
                        stock = int(line[6:])
                    else:
                        phase +=1

                # Phase 1: Read in kitchen map.
                elif phase == 1:
                    for x, rep in enumerate(line):
                        # Object, i.e. Tomato, Lettuce, Onion, or Plate.
                        if rep in 'tlop':
                            # creates as many of item as needed at counter location
                            counter = SpawnCounter(location=(x, y))
                            for i in range (0, stock):
                                obj = Object(
                                        location=(x, y),
                                        contents=RepToClass[rep]())
                                counter.acquire(obj=obj)
                                self.world.insert(obj)
                            self.world.insert(obj=counter)
                        # GridSquare, i.e. Floor, Counter, Cutboard, Delivery.
                        elif rep in RepToClass:
                            newobj = RepToClass[rep]((x, y))
                            self.world.objects.setdefault(newobj.name, []).append(newobj)
                        else:
                            # Empty. Set a Floor tile.
                            f = Floor(location=(x, y))
                            self.world.objects.setdefault('Floor', []).append(f)
                    y += 1
                # Phase 2: Read in recipe list.
                elif phase == 2:
                    self.recipes.append(globals()[line]())

                # Phase 3: Read whether teams (competitive) mode or coop mode
                elif phase == 3:
                    if 'teams' in line:
                        phase = 4
                    else:
                        phase = 5

                # Phase 4: Read in agent locations (up to num_agents) for agents on teams.
                elif phase == 4:
                    # if level is designed for teams
                    if len(self.sim_agents) < num_agents:
                        loc = line.split(' ')
                        sim_agent = SimAgent(
                            name='agent-' + str(len(self.sim_agents) + 1),
                            id_color=TEAM_COLORS[len(self.sim_agents) % 2][int(len(self.sim_agents)/2)],
                            location=(int(loc[0]), int(loc[1])))
                        team = (len(self.sim_agents) % 2) + 1
                        sim_agent.set_team(team)
                        if team == 1:
                            self.team1_agents.append('agent-' + str(len(self.sim_agents) + 1))
                        else:
                            self.team2_agents.append('agent-' + str(len(self.sim_agents) + 1))
                        self.sim_agents.append(sim_agent)

                elif phase == 5:
                    if len(self.sim_agents) < num_agents:
                        loc = line.split(' ')
                        sim_agent = SimAgent(
                            name='agent-'+str(len(self.sim_agents)+1),
                            id_color=COLORS[len(self.sim_agents)],
                            location=(int(loc[0]), int(loc[1])))
                        self.sim_agents.append(sim_agent)
        
        self.distances = {}
        self.world.width = x+1
        self.world.height = y
        self.world.perimeter = 2*(self.world.width + self.world.height)

        # self.observation_space = []
        self.observation_space = (Box(low=0, high=2, shape=(self.world.height, self.world.width), dtype=np.int32))

    # def create_spaces(self):
    #     # observation space - locations of special gridsquares (blue, red, trash), objects and agents in the world
    #     self.observation_space = spaces.Box(0, self.world.width), spaces.Box(0, self.world.height)
        # # self.observation_space = 
        # self.action_space = spaces.Discrete(5)

    def reset(self):
        self.world = World(arglist=self.arglist)
        self.recipes = []
        self.sim_agents = []
        self.agent_actions = {}
        self.t = 0

        # For visualizing episode.
        self.rep = []

        # For tracking data during an episode.
        self.collisions = []
        self.termination_info = ""
        self.successful = False

        # Load world & distances.
        self.load_level(
                level=self.arglist.level,
                num_agents=self.arglist.num_agents)
        self.all_subtasks = self.run_recipes()
        # self.confrontation_tasks()
        self.world.make_loc_to_gridsquare()
        self.world.make_reachability_graph()
        self.cache_distances()
        self.obs_tm1 = copy.copy(self)

        if self.arglist.record or self.arglist.with_image_obs:
            self.game = GameImage(
                    filename=self.filename,
                    world=self.world,
                    sim_agents=self.sim_agents,
                    record=self.arglist.record,
                    env=self)
            self.game.on_init()
            if self.arglist.record:
                self.game.save_image_obs(self.t)
        else:
            self.game = None

        # return self.create_obs()
        return copy.copy(self)

    def close(self):
        return
    

    def create_obs(self):
        # make a simple vector containing location of all agents, delivery locations
        obs = []
        # for agent in self.sim_agents:
        #     obs.append(agent.location)
        
        # delivery_1 = list(filter(lambda o: o.name=='DeliveryBlue', self.world.get_object_list()))[0].location
        # delivery_2 = list(filter(lambda o: o.name=='DeliveryRed', self.world.get_object_list()))[0].location
        # obs.append(delivery_1)
        # obs.append(delivery_2)

        self.update_display()

        # get string representation of objects in the world

        string_rep = self.world.string_rep

        for agent in self.sim_agents:
            string_rep[agent.location[1]][agent.location[0]] = 'a'

        for row in string_rep:
            for elem in row:
                obs.append(rep_to_int(elem))

        return obs

    def step(self, action_dict, agents):
        # Track internal environment info.
        self.t += 1
        print("===============================")
        print("[environment.step] @ TIMESTEP {}".format(self.t))
        print("===============================")

        # obs2 = []
        # Get actions.
        for sim_agent in self.sim_agents:
            sim_agent.action = action_dict[sim_agent.name]

        # Check collisions.
        self.check_collisions()
        self.obs_tm1 = copy.copy(self)

        # Execute.
        self.execute_navigation()

        # get locations of agents after performing actions
        # for sim_agent in self.sim_agents:
        #     obs2.append(sim_agent.location)

        # Visualize.
        # self.display()
        # self.print_agents()
        if self.arglist.record:
            self.game.save_image_obs(self.t)

        # Get a plan-representation observation.
        new_obs = copy.copy(self)
        # Get an image observation
        if self.game != None:
            image_obs = self.game.get_image_obs()

        done = self.done()

        reward1, reward2 = self.reward(self.obs_tm1, new_obs, agents)

        # obs2 = self.create_obs()

        info = {"t": self.t, "obs": new_obs,
                "image_obs": image_obs,
                "done": done, "termination_info": self.termination_info}
        return new_obs, reward1, reward2, done, info
        # return obs2, reward1, reward2, done, info

    def get_team1_score(self):
        return self.team1_score

    def increase_team1_score(self, score):
        self.team1_score +=score

    def get_team2_score(self):
        return self.team2_score

    def increase_team2_score(self, score):
        self.team2_score += score

    def done(self):
        # Done if the episode maxes out
        if self.t >= self.arglist.max_num_timesteps and self.arglist.max_num_timesteps:
            self.termination_info = "Terminating because passed {} timesteps".format(
                    self.arglist.max_num_timesteps)
            self.successful = False
            return True
        return False

    

    def picked_up(self, agents):
        for agent in agents:
            print("hi")

    def delivered(self):
        for subtask in self.all_subtasks:
        # Double check all goal_objs are at Delivery.
            if isinstance(subtask, recipe.Deliver):
                _, goal_obj = nav_utils.get_subtask_obj(subtask)

                delivery_1 = list(filter(lambda o: o.name=='DeliveryBlue', self.world.get_object_list()))[0].location
                delivery_2 = list(filter(lambda o: o.name=='DeliveryRed', self.world.get_object_list()))[0].location
                goal_obj_locs = self.world.get_all_object_locs(obj=goal_obj)
                delivered_1 = len(list(filter(lambda a: a in delivery_1, goal_obj_locs)))
                delivered_2 = len(list(filter(lambda a: a in delivery_2, goal_obj_locs)))

        # returns whether there was a new delivery for each team in this timestep
        return delivered_1 > self.delivered1, delivered_2 > self.delivered2
            
    
    def reward(self, old_obs, new_obs, agents):
       # rewards for each team depending if something was delivered
        r1, r2 = 0, 0
        score1, score2 = self.delivered()
        if score1:    # if team1 delivered, increases their score
            r1 += 10
            r2 += -10
        elif score2:     # if team2 delivered, increases their score
            r1 += -10
            r2 += 10

        # get reward for each agent completing a subtask, then combine rewards for agents on the same team
        for agent in agents:
            if agent.team == 1:
                r1 += agent.get_reward(old_obs, new_obs)
            elif agent.team == 2:
                r2 += agent.get_reward(old_obs, new_obs)

        
        # return two rewards (team1, team2)
        return r1, r2

    def print_agents(self):
        for sim_agent in self.sim_agents:
            sim_agent.print_status()

    def display(self):
        self.update_display()
        print(str(self))

    def update_display(self):
        self.rep = self.world.update_display()
        for agent in self.sim_agents:
            x, y = agent.location
            # self.rep[y][x] = agent.name
            self.rep[y][x] = 'a'

    def get_agent_names(self):
        return [agent.name for agent in self.sim_agents]
    
    def get_agent_team(self, agent_name):
        if agent_name in self.team1_agents:
            return 1
        elif agent_name in self.team2_agents:
            return 2

    def run_recipes(self):
        """Returns different permutations of completing recipes."""
        self.sw = STRIPSWorld(world=self.world, recipes=self.recipes)
        # [path for recipe 1, path for recipe 2, ...] where each path is a list of actions
        subtasks = self.sw.get_subtasks(max_path_length=self.arglist.max_num_subtasks)
        all_subtasks = [subtask for path in subtasks for subtask in path]

        # adding confrontation tasks to possible subtasks
        all_subtasks += self.recipes[0].get_con_actions()
        print("HERE ARE ALL SUBTASKS: ", all_subtasks)
        return all_subtasks
        
    def get_AB_locs_given_objs(self, agent, subtask, subtask_agent_names, start_obj, goal_obj, subtask_action_obj):
        """Returns list of locations relevant for subtask's Merge operator.

        See Merge operator formalism in our paper, under Fig. 11:
        https://arxiv.org/pdf/2003.11778.pdf"""

        # For Merge operator on Chop subtasks, we look at objects that can be
        # chopped and the cutting board objects.
        if isinstance(subtask, recipe.Chop):
            # A: Object that can be chopped.
            A_locs = self.world.get_object_locs(obj=start_obj, is_held=False) + list(map(lambda a: a.location,\
                list(filter(lambda a: a.name in subtask_agent_names and a.holding == start_obj, self.sim_agents))))

            # B: Cutboard objects.
            B_locs = self.world.get_all_object_locs(obj=subtask_action_obj)

        # For Merge operator on Deliver subtasks, we look at objects that can be
        # delivered and the Delivery object.
        elif isinstance(subtask, recipe.Deliver):
            # B: Corresponding delvery locations
            if agent.team == 1:
                blue_delivery = nav_utils.get_obj(obj_string="DeliveryBlue", type_="is_supply", state=None)
                B_locs = self.world.get_all_object_locs(obj=blue_delivery)
                
            elif agent.team == 2:
                red_delivery = nav_utils.get_obj(obj_string="DeliveryRed", type_="is_supply", state=None)
                B_locs = self.world.get_all_object_locs(obj=red_delivery)

            # A: Object that can be delivered.
            A_locs = self.world.get_object_locs(
                    obj=start_obj, is_held=False) + list(
                            map(lambda a: a.location, list(
                                filter(lambda a: a.name in subtask_agent_names and a.holding == start_obj, self.sim_agents))))
            A_locs = list(filter(lambda a: a not in B_locs, A_locs))

            # print("Want to put ingredients", start_obj, "at A_locs", A_locs, "to", B_locs)

        # For Merge operator on Merge subtasks, we look at objects that can be
        # combined together. These objects are all ingredient objects (e.g. Tomato, Lettuce).
        elif isinstance(subtask, recipe.Merge):
            A_locs = self.world.get_object_locs(
                    obj=start_obj[0], is_held=False) + list(
                            map(lambda a: a.location, list(
                                filter(lambda a: a.name in subtask_agent_names and a.holding == start_obj[0], self.sim_agents))))
            B_locs = self.world.get_object_locs(
                    obj=start_obj[1], is_held=False) + list(
                            map(lambda a: a.location, list(
                                filter(lambda a: a.name in subtask_agent_names and a.holding == start_obj[1], self.sim_agents))))

        # for Merge operator on Hoard subtasks, we look at empty counters that are near to agents on team
        elif isinstance(subtask, recipe.Hoard):
            # Ingredients to hoard
            if agent.team == 1:
                agents = self.team1_agents
            elif agent.team == 2:
                agents = self.team2_agents
            
            # Where to put hoarded ingredients (near team agents)
            
            # finds location of other agents on this agent's team
            agent_locs = list(map(lambda a: a.location, list(filter(lambda a: agent.team == a.team, self.sim_agents))))     

            # find location of empty counters near team agents
            counter_locs = self.world.get_all_object_locs(obj=subtask_action_obj)
            nearby_locs = []
            for team_agent in agent_locs:
                nearby_locs += list(filter(lambda a: abs(team_agent[0]-a[0]) < 3 and abs(team_agent[1]-a[1]) < 3, counter_locs))
            
            B_locs = list(filter(lambda a: self.world.get_gridsquare_at(a).free(), nearby_locs))

            max_x, max_y = self.world.width-1, self.world.height-1

            unreachable = [(0, 0), (max_x, 0), (0, max_y), (max_x, max_y)]
            # cannot access counters at the corner of maps, so remove these 
            B_locs = list(filter(lambda a: a not in unreachable, B_locs))
            
            if len(B_locs) == 0:   
                B_locs = agent_locs

            B_locs = [(4, 0)]

            A_locs = self.world.get_object_locs(
                    obj=goal_obj, is_held=False) + list(
                            map(lambda a: a.location, list(
                                filter(lambda a: a.name in agents and a.holding == start_obj, self.sim_agents))))
            A_locs = set((filter(lambda a: a not in B_locs, A_locs)))

            # print("Want to put ingredients", start_obj, "at A_locs", A_locs, "to", B_locs, "because I'm currently holding", agent.holding)


        # for Merge operator on Trash subtasks, we look at trashcan spaces and put whatever the agent is holding there
        elif isinstance(subtask, recipe.Trash):
            # locations of trashcans
            B_locs = self.world.get_all_object_locs(obj=subtask_action_obj)

            # locations of objects to trash
            A_locs = self.world.get_object_locs(
                    obj=start_obj, is_held=False) + list(
                            map(lambda a: a.location, list(
                                filter(lambda a: a.name in subtask_agent_names and a.holding == start_obj, self.sim_agents))))

            A_locs = list(filter(lambda a: a not in B_locs, A_locs)) 
            
        # for Merge operator on Steal subtasks, we look for dishes last held by the other team and put them closer to our team's agents
        elif isinstance(subtask, recipe.Steal):
            
            # need to get dish objects to check who last held them
            dishes = map(lambda d: self.world.get_object_at(d, None, find_held_objects = False), self.world.get_object_locs(obj=goal_obj, is_held=False))
            # filter to get locations of dishes that were last held by the other team

            A_locs = list(map(lambda l: l.get_location(), list(filter(lambda a: (a.last_held != agent.team), dishes))))

            # locations near this team's agents
            agent_locs = list(map(lambda a: a.location, list(filter(lambda a: agent.team == a.team, self.sim_agents))))         

            # find location of empty counters near team agents
            counter_locs = self.world.get_all_object_locs(obj=subtask_action_obj)
            for team_agent in agent_locs:
                nearby_locs = (list(filter(lambda a: abs(team_agent[0]-a[0]) < 3 and abs(team_agent[1]-a[1]) < 3, counter_locs)))
            
            B_locs = list(filter(lambda a: self.world.get_gridsquare_at(a).free(), nearby_locs))


        else:
            return [], []
        return A_locs, B_locs

    def get_lower_bound_for_subtask_given_objs(
            self, subtask, subtask_agent_names, start_obj, goal_obj, subtask_action_obj):
        """Return the lower bound distance (shortest path) under this subtask between objects."""
        # assert len(subtask_agent_names) <= 4, 'passed in {} agents but can only do 1 or 2'.format(len(agents))

        # Calculate extra holding penalty if the object is irrelevant.
        holding_penalty = 0.0
        for agent in self.sim_agents:
            if agent.name in subtask_agent_names:
                # Check for whether the agent is holding something.
                if agent.holding is not None:
                    if isinstance(subtask, recipe.Merge):
                        continue
                    else:
                        if agent.holding != start_obj and agent.holding != goal_obj:
                            # Add one "distance"-unit cost
                            holding_penalty += 2.0
        # Account for two-agents where we DON'T want to overpenalize.
        holding_penalty = min(holding_penalty, 2)

        # Get current agent locations.
        agent_locs = [agent.location for agent in list(filter(lambda a: a.name in subtask_agent_names, self.sim_agents))]
        A_locs, B_locs = self.get_AB_locs_given_objs(
                subtask=subtask,
                agent=self.sim_agents[0],
                subtask_agent_names=subtask_agent_names,
                start_obj=start_obj,
                goal_obj=goal_obj,
                subtask_action_obj=subtask_action_obj)
        
        # Add together distance and holding_penalty.
        return self.world.get_lower_bound_between(
                subtask=subtask,
                agent_locs=tuple(agent_locs),
                A_locs=tuple(A_locs),
                B_locs=tuple(B_locs)) + holding_penalty

    def is_collision(self, agent1_loc, agent2_loc, agent1_action, agent2_action):
        """Returns whether agents are colliding.

        Collisions happens if agent collide amongst themselves or with world objects."""
        # Tracks whether agents can execute their action.
        execute = [True, True]

        # Collision between agents and world objects.
        agent1_next_loc = tuple(np.asarray(agent1_loc) + np.asarray(agent1_action))
        if self.world.get_gridsquare_at(location=agent1_next_loc).collidable:
            # Revert back because agent collided.
            agent1_next_loc = agent1_loc

        agent2_next_loc = tuple(np.asarray(agent2_loc) + np.asarray(agent2_action))
        if self.world.get_gridsquare_at(location=agent2_next_loc).collidable:
            # Revert back because agent collided.
            agent2_next_loc = agent2_loc

        
        # Inter-agent collision.
        if agent1_next_loc == agent2_next_loc:
            if agent1_next_loc == agent1_loc and agent1_action != (0, 0):
                execute[1] = False
            elif agent2_next_loc == agent2_loc and agent2_action != (0, 0):
                execute[0] = False
            else:
                execute[0] = False
                execute[1] = False

        # Prevent agents from swapping places.
        elif ((agent1_loc == agent2_next_loc) and
                (agent2_loc == agent1_next_loc)):
            execute[0] = False
            execute[1] = False
        return execute

    def check_collisions(self):
        """Checks for collisions and corrects agents' executable actions.

        Collisions can either happen amongst agents or between agents and world objects."""
        execute = [True for _ in self.sim_agents]

        # Check each pairwise collision between agents.
        for i, j in combinations(range(len(self.sim_agents)), 2):
            agent_i, agent_j = self.sim_agents[i], self.sim_agents[j]
            exec_ = self.is_collision(
                    agent1_loc=agent_i.location,
                    agent2_loc=agent_j.location,
                    agent1_action=agent_i.action,
                    agent2_action=agent_j.action)

            # Update exec array and set path to do nothing.
            if not exec_[0]:
                execute[i] = False
            if not exec_[1]:
                execute[j] = False

            # Track collisions.
            if not all(exec_):
                collision = CollisionRepr(
                        time=self.t,
                        agent_names=[agent_i.name, agent_j.name],
                        agent_locations=[agent_i.location, agent_j.location])
                self.collisions.append(collision)

        print('\nexecute array is:', execute)

        # Update agents' actions if collision was detected.
        for i, agent in enumerate(self.sim_agents):
            if not execute[i]:
                agent.action = (0, 0)
            print("{} at {} has action {}".format((agent.name, agent.color), agent.location, agent.action))

    def execute_navigation(self):
        for agent in self.sim_agents:
            returnVal = interact(agent=agent, world=self.world)
            if returnVal == 1:      # an agent on team1 delivered a dish
                self.delivered1 += 1
                self.team1_score += 100
            elif returnVal == 2:      # an agent on team2 delivered a dish
                self.delivered2 += 1
                self.team2_score += 100
            self.agent_actions[agent.name] = agent.action


    def cache_distances(self):
        """Saving distances between world objects."""
        counter_grid_names = [name for name in self.world.objects if "Supply" in name or "Counter" in name or "Delivery" in name or "Cut" in name]
        # Getting all source objects.
        source_objs = copy.copy(self.world.objects["Floor"])
        for name in counter_grid_names:
            source_objs += copy.copy(self.world.objects[name])
        # Getting all destination objects.
        dest_objs = source_objs

        # From every source (Counter and Floor objects),
        # calculate distance to other nodes.
        for source in source_objs:
            self.distances[source.location] = {}
            # Source to source distance is 0.
            self.distances[source.location][source.location] = 0
            for destination in dest_objs:
                # Possible edges to approach source and destination.
                source_edges = [(0, 0)] if not source.collidable else World.NAV_ACTIONS
                destination_edges = [(0, 0)] if not destination.collidable else World.NAV_ACTIONS
                # Maintain shortest distance.
                shortest_dist = np.inf
                for source_edge, dest_edge in product(source_edges, destination_edges):
                    try:
                        dist = nx.shortest_path_length(self.world.reachability_graph, (source.location,source_edge), (destination.location, dest_edge))
                        # Update shortest distance.
                        if dist < shortest_dist:
                            shortest_dist = dist
                    except:
                        continue
                # Cache distance floor -> counter.
                self.distances[source.location][destination.location] = shortest_dist

        # Save all distances under world as well.
        self.world.distances = self.distances