# Recipe planning
from recipe_planner.stripsworld import STRIPSWorld
import recipe_planner.utils as recipe_utils
from recipe_planner.utils import *

# Delegation planning
from delegation_planner.bayesian_delegator import BayesianDelegator

# Navigation planner
from navigation_planner.planners.e2e_brtdp import E2E_BRTDP
import navigation_planner.utils as nav_utils

# Other core modules
from utils.core import Counter, Cutboard, Food, FoodState
from utils.utils import agent_settings

import numpy as np
import copy
from termcolor import colored as color
from collections import namedtuple

AgentRepr = namedtuple("AgentRepr", "name location holding")

# Colors for agents.
COLORS = ['blue', 'magenta', 'yellow', 'green']
TEAM_COLORS = [['blue-team-blue', 'magenta-team-blue', 'yellow-team-blue', 'green-team-blue'], ['cat-blue-team-red', 'cat-grey-team-red', 'cat-orange-team-red', 'cat-yellow-team-red']]


class RealAgent:
    """Real Agent object that performs task inference and plans."""

    def __init__(self, arglist, name, id_color, recipes, hoarder):
        self.arglist = arglist
        self.name = name
        self.color = id_color
        self.recipes = recipes
        
        self.team = 1  # what team agent is on - cooperates with same team, opponents with diff. teams - default team is 1 (blue)

        # Bayesian Delegation.
        self.hoarder = hoarder   # Boolean - true if this agent's role is to hoard ingredients

        self.reset_subtasks()
        self.new_subtask = None
        self.new_subtask_agent_names = []
        self.incomplete_subtasks = []
        self.signal_reset_delegator = False
        self.is_subtask_complete = lambda w: False
        self.beta = arglist.beta
        self.none_action_prob = 0.5

        self.model_type = agent_settings(arglist, name)
        if self.model_type == "up":
            self.priors = 'uniform'
        else:
            self.priors = 'spatial'

        # Navigation planner.
        self.planner = E2E_BRTDP(
                alpha=arglist.alpha,
                tau=arglist.tau,
                cap=arglist.cap,
                main_cap=arglist.main_cap)

    def __str__(self):
        return (self.name[-1], self.color)

    def __copy__(self):
        a = RealAgent(arglist=self.arglist,
                name=self.name,
                id_color=self.color,
                recipes=self.recipes)
        a.subtask = self.subtask
        a.new_subtask = self.new_subtask
        a.subtask_agent_names = self.subtask_agent_names
        a.new_subtask_agent_names = self.new_subtask_agent_names
        a.__dict__ = self.__dict__.copy()
        if self.holding is not None:
            a.holding = copy.copy(self.holding)
        return a

    def set_team(self, team):
        self.team = team

    def get_team(self):
        return self.team

    def get_holding(self):
        if self.holding is None:
            return 'None'
        return self.holding.full_name

    def select_action(self, obs):
        """Return best next action for this agent given observations."""
        sim_agent = list(filter(lambda x: x.name == self.name, obs.sim_agents))[0]
        self.location = sim_agent.location
        self.holding = sim_agent.holding
        self.action = sim_agent.action

        if obs.t == 0:
            self.setup_subtasks(env=obs)

        # Select subtask based on Bayesian Delegation.
        self.update_subtasks(env=obs)
        self.new_subtask, self.new_subtask_agent_names = self.delegator.select_subtask(
            agent_name=self.name)

        # !!! if new_subtask is hoard subtask, need to get object first 
        # if isinstance(self.new_subtask, recipe_utils.Hoard):
        #     print(self.holding, "Is chopped: ", self.holding.is_chopped())
        #     if self.holding is None or not self.holding.is_chopped():
        #         self.new_subtask = recipe_utils.Chop(self.new_subtask.args[0])
        #         print("Want to do hoard so first need to ", self.new_subtask)     
        #         if recipe_utils.Hoard(self.new_subtask.args[0]) not in self.incomplete_subtasks:
        #             self.incomplete_subtasks.append(recipe_utils.Hoard(self.new_subtask.args[0]))
            
        self.plan(copy.copy(obs))
        return self.action

    def get_subtasks(self, world):
        """Return different subtask permutations for recipes."""
        self.sw = STRIPSWorld(world, self.recipes)
        # [path for recipe 1, path for recipe 2, ...] where each path is a list of actions.
        subtasks = self.sw.get_subtasks(max_path_length=self.arglist.max_num_subtasks)
        all_subtasks = []
        
        # add subtasks for as many ingredients as there are available in the world
       
        all_subtasks += [subtask for path in subtasks for subtask in path]
        all_subtasks += self.recipes[0].get_con_actions()

        print("Getting subtasks agent can perform: ", all_subtasks)
        # Uncomment below to view graph for recipe path i
        # i = 0
        # pg = recipe_utils.make_predicate_graph(self.sw.initial, recipe_paths[i])
        # ag = recipe_utils.make_action_graph(self.sw.initial, recipe_paths[i])
        return all_subtasks

    def setup_subtasks(self, env):
        """Initializing subtasks and subtask allocator, Bayesian Delegation."""
        self.incomplete_subtasks = []

        self.ingredients = self.recipes[0].get_ingredients()
        self.ingredients.append("Plate")
        if self.hoarder:
            to_hoard = self.recipes[0].get_ingredients()
            for ingredient in to_hoard:
                if len(env.world.get_object_locs(obj=ingredient, is_held=False)) == 0:
                    self.incomplete_subtasks.append(recipe_utils.Hoard(ingredient.get_name()))
        
        self.incomplete_subtasks += self.get_subtasks(world=env.world)
        self.delegator = BayesianDelegator(
                agent_name=self.name,
                all_agent_names=env.get_agent_names(),
                model_type=self.model_type,
                planner=self.planner,
                none_action_prob=self.none_action_prob, team=self.team)

    def reset_subtasks(self):
        """Reset subtasks---relevant for Bayesian Delegation."""
        self.subtask = None
        self.subtask_agent_names = []
        self.subtask_complete = False

    def refresh_subtasks(self, world):
        """Refresh subtasks---relevant for Bayesian Delegation."""
        # Check whether subtask is complete.
        self.subtask_complete = False
        if self.subtask is None or len(self.subtask_agent_names) == 0:
            print("{} has no subtask".format((self.name, self.color)))
            return
        self.subtask_complete = self.is_subtask_complete(world)
        print("{} done with {} according to planner: {}\nplanner has subtask {} with subtask object {}".format(
            (self.name, self.color),
            self.subtask, self.is_subtask_complete(world),
            self.planner.subtask, self.planner.goal_obj))

        # Refresh for incomplete subtasks.
        if self.subtask_complete:
            if self.subtask in self.incomplete_subtasks:
                self.incomplete_subtasks.remove(self.subtask)
                self.subtask_complete = True
            # if no more subtasks and there is stock left in the world, re-add chop-merge-deliver subtasks
            if len(self.incomplete_subtasks) == 1:
                print("Gonna reset subtasks now")
                keep_cooking = True
                for ingredient in self.ingredients:
                    if ingredient == "Plate":
                        obj = nav_utils.get_obj(obj_string="Plate", type_="is_object", state=None)
                    else:
                        obj = nav_utils.get_obj(obj_string=ingredient, type_="is_object", state=FoodState.FRESH)
                    print("Stock of ingredient", ingredient, "is", len(world.get_object_locs(obj=obj, is_held=False)))
                    if len(world.get_object_locs(obj=obj, is_held=False)) == 0:
                        keep_cooking = False
                if keep_cooking:
                    self.reset_subtasks()
                    self.incomplete_subtasks = self.get_subtasks(world)
        print('{} incomplete subtasks:'.format(
            (self.name, self.color)),
            ', '.join(str(t) for t in self.incomplete_subtasks))

    def update_subtasks(self, env):
        """Update incomplete subtasks---relevant for Bayesian Delegation."""
        if ((self.subtask is not None and self.subtask not in self.incomplete_subtasks)
                or (self.delegator.should_reset_priors(obs=copy.copy(env),
                                                       incomplete_subtasks=self.incomplete_subtasks))):
            self.reset_subtasks()
            self.delegator.set_priors(
                obs=copy.copy(env),
                incomplete_subtasks=self.incomplete_subtasks,
                priors_type=self.priors)
        else:
            if self.subtask is None:
                self.delegator.set_priors(
                    obs=copy.copy(env),
                    incomplete_subtasks=self.incomplete_subtasks,
                    priors_type=self.priors)
            else:
                self.delegator.bayes_update(
                    obs_tm1=copy.copy(env.obs_tm1),
                    actions_tm1=env.agent_actions,
                    beta=self.beta)

    def all_done(self):
        """Return whether this agent is all done.
        An agent is done if all Deliver subtasks are completed."""
        if any([isinstance(t, Deliver) for t in self.incomplete_subtasks]):
            return False
        return True

    def get_action_location(self):
        """Return location if agent takes its action---relevant for navigation planner."""
        return tuple(np.asarray(self.location) + np.asarray(self.action))

    def plan(self, env, initializing_priors=False):
        """Plan next action---relevant for navigation planner."""
        print('right before planning, {} had old subtask {}, new subtask {}, subtask complete {}'.format(self.name, self.subtask, self.new_subtask, self.subtask_complete))

        # Check whether this subtask is done.
            # if subtask is done, update to next one
        if self.new_subtask is not None:
            self.def_subtask_completion(env=env)

        # If subtask is None, then do nothing.
        if (self.new_subtask is None) or (not self.new_subtask_agent_names):
            actions = nav_utils.get_single_actions(env=env, agent=self)
            probs = []
            for a in actions:
                if a == (0, 0):
                    probs.append(self.none_action_prob)
                else:
                    probs.append((1.0-self.none_action_prob)/(len(actions)-1))
            self.action = actions[np.random.choice(len(actions), p=probs)]
        # Otherwise, plan accordingly.
        else:
            if self.model_type == 'greedy' or initializing_priors:
                other_agent_planners = {}
            else:
                # Determine other agent planners for level 1 planning.
                # Other agent planners are based on your planner---agents never
                # share planners.
                backup_subtask = self.new_subtask if self.new_subtask is not None else self.subtask
                other_agent_planners = self.delegator.get_other_agent_planners(
                        obs=copy.copy(env), backup_subtask=backup_subtask)

            print("[ {} Planning ] Task: {}, Task Agents: {}".format(
                self.name, self.new_subtask, self.new_subtask_agent_names))

            action = self.planner.get_next_action(
                    env=env, subtask=self.new_subtask,
                    subtask_agent_names=self.new_subtask_agent_names,
                    other_agent_planners=other_agent_planners, team=self.team)

            # If joint subtask, pick your part of the simulated joint plan.
            if self.name not in self.new_subtask_agent_names and self.planner.is_joint:
                self.action = action[0]
            else:
                self.action = action[self.new_subtask_agent_names.index(self.name)] if self.planner.is_joint else action

        # Update subtask.
        self.subtask = self.new_subtask
        self.subtask_agent_names = self.new_subtask_agent_names
        self.new_subtask = None
        self.new_subtask_agent_names = []

        print('{} proposed action: {}\n'.format(self.name, self.action))

    def def_subtask_completion(self, env):
        # Determine desired objects.
        self.start_obj, self.goal_obj = nav_utils.get_subtask_obj(subtask=self.new_subtask)
        self.subtask_action_object = nav_utils.get_subtask_action_obj(subtask=self.new_subtask, team=self.team)

        # Define termination conditions for agent subtask.
        # For Deliver subtask, desired object should be at a Deliver location.
        if isinstance(self.new_subtask, Deliver):
            self.cur_obj_count = len(list(
                filter(lambda o: o in set(env.world.get_all_object_locs(self.subtask_action_object)),
                env.world.get_object_locs(obj=self.goal_obj, is_held=False))))
            self.has_more_obj = lambda x: int(x) > self.cur_obj_count
            self.is_subtask_complete = lambda w: self.has_more_obj(
                    len(list(filter(lambda o: o in
                set(env.world.get_all_object_locs(obj=self.subtask_action_object)),
                w.get_object_locs(obj=self.goal_obj, is_held=False)))))
            
        # For Trash subtask, remove object from world so lower number of goal objects
        elif isinstance(self.new_subtask, Trash):

            # gets count of all goal objects that haven't already been delivered
            self.cur_obj_count = len(list(env.world.get_all_object_locs(self.goal_obj)))
                # but can't trash delivered objects - remove delivered objects?

            self.has_less_obj = lambda x: int(x) < self.cur_obj_count
            self.is_subtask_complete = lambda w: self.has_less_obj(
                    len(w.get_all_object_locs(self.goal_obj)))

        elif isinstance(self.new_subtask, Hoard):
            
            # gets number of ingredients currently in world (only those on counters, not including multiples stocked at spawn)
            self.cur_obj_count = len(set(env.world.get_object_locs(self.goal_obj, is_held=False)))
            print("Current count:", self.cur_obj_count)

            self.has_more_obj = lambda x: int(x) > self.cur_obj_count
            

            # self.is_subtask_complete = lambda w: self.has_more_obj(
            #     len(list(filter(lambda o: o in set(w.get_all_object_locs(self.subtask_action_obj)),
            #     w.get_object_locs(obj=self.goal_obj, is_held=False)))))

            self.is_subtask_complete = lambda w: self.has_more_obj(len(set(w.get_object_locs(self.goal_obj, is_held=False))))


        elif isinstance(self.new_subtask, Steal):
            # all dishes - excluding those that have already been delivered
            dishes = map(lambda d: env.world.get_object_at(d, None, find_held_objects = False), env.world.get_object_locs(obj=self.goal_obj, is_held=False))

            self.cur_obj_count = len(list(filter(lambda a: a.last_held != self.team, dishes)))

            self.has_less_obj = lambda x: int(x) < self.cur_obj_count
            # if self.removed_object is not None and self.removed_object == self.goal_obj:
            #     self.is_subtask_complete = lambda w: self.has_less_obj(
            #             len(w.get_all_object_locs(self.goal_obj)) + 1)
            # else:
            self.is_subtask_complete = lambda w: self.has_less_obj(
                   len(list(filter(lambda a: (a.last_held != self.team), map(lambda d: w.get_object_at(d, None, find_held_objects = False), w.get_object_locs(obj=self.goal_obj, is_held=False))))))

            # dishes = list(map(lambda x: env.world.get_objects_at(x), env.world.get_all_object_locs(self.goal_obj)))
            # if len(dishes) > 0:
            #     self.cur_obj_count = len(list(filter(lambda a: a.last_held != self.team, dishes)))
            #     print("count of objects last held by the other team")
            # else:
            #     self.cur_obj_count = 0 

            # # successful if count of dishes held by last team goes down
            # self.has_less_obj = lambda x: int(x) < self.cur_obj_count

            # self.is_subtask_complete = lambda w: self.has_less_obj(
            #         len(list(filter(lambda a: a is not None and a.last_held != self.team, map(lambda x: w.get_objects_at(x), w.get_all_object_locs(self.goal_obj))))))

        # Otherwise, for other subtasks, check based on # of objects attributed to your team
        else:
            self.cur_obj_count = len(env.world.get_all_object_locs(obj=self.goal_obj))

            # Goal state is reached when the number of desired objects has increased.
            self.is_subtask_complete = lambda w: len(w.get_all_object_locs(obj=self.goal_obj)) > self.cur_obj_count
            # self.is_subtask_complete = lambda w: len(list(filter(lambda b: b is not None and b.last_held == self.team, map(lambda a: w.get_objects_at(a)[0], env.world.get_all_object_locs(obj=self.goal_obj))))) > self.cur_obj_count

class SimAgent:
    """Simulation agent used in the environment object."""

    def __init__(self, name, id_color, location):
        self.name = name
        self.color = id_color
        self.location = location
        self.holding = None
        self.action = (0, 0)
        self.has_delivered = False
        self.team = 1

    def __str__(self):
        return self.color
        # return color(self.name[-1], self.color)

    def __copy__(self):
        a = SimAgent(name=self.name, id_color=self.color,
                location=self.location)
        a.__dict__ = self.__dict__.copy()
        if self.holding is not None:
            a.holding = copy.copy(self.holding)
        return a

    def set_team(self, team):
        self.team = team

    def get_team(self):
        return self.team

    def get_repr(self):
        return AgentRepr(name=self.name, location=self.location, holding=self.get_holding())

    def get_holding(self):
        if self.holding is None:
            return 'None'
        return self.holding.full_name

    def print_status(self):
        print("{} currently at {}, action {}, holding {}".format(
                self.color,
                self.location,
                self.action,
                self.get_holding()))

    def acquire(self, obj):
        if self.holding is None:
            self.holding = obj
            self.holding.is_held = True
            self.holding.location = self.location
        else:
            self.holding.merge(obj) # Obj(1) + Obj(2) => Obj(1+2)

    def release(self):
        self.holding.is_held = False
        self.holding = None

    def move_to(self, new_location):
        self.location = new_location
        if self.holding is not None:
            self.holding.location = new_location
        
    def get_location(self):
        return self.location
