from utils.core import *
import numpy as np

def interact(agent, world):
    """Carries out interaction for this agent taking this action in this world.

    The action that needs to be executed is stored in `agent.action`.
    """

    # agent does nothing (i.e. no arrow key)
    if agent.action == (0, 0):
        return

    action_x, action_y = world.inbounds(tuple(np.asarray(agent.location) + np.asarray(agent.action)))
    # action_x, action_y = world.inbounds((agent.location[0]+agent.action[0][0], agent.location[1]+agent.action[0][1]))

    gs = world.get_gridsquare_at((action_x, action_y))

    # if floor in front --> move to that square
    if isinstance(gs, Floor): #and gs.holding is None:
        agent.move_to(gs.location)

    # if holding something
    elif agent.holding is not None:
        # if delivery in front --> deliver
        if isinstance(gs, Delivery):
            # removes any dishes already there
            obj = agent.holding
            if obj.is_deliverable():
                # already_del = world.get_object_at(gs.location, None, find_held_objects=False)
                # if already_del is not None:
                #     world.remove(already_del)
                gs.acquire(obj)
                agent.release()
                # world.remove(obj)
                # returns number of corresponding team to increase their score
                if isinstance(gs, DeliveryBlue):
                    return 1
                elif isinstance(gs, DeliveryRed):
                    return 2

        # if trashcan in front --> delete object from world
        elif isinstance(gs, Trashcan):
            obj = agent.holding
            world.remove(obj)
            agent.release()

        # if occupied gridsquare in front --> try merging
        elif world.is_occupied(gs.location):
            # Get object on gridsquare/counter
            obj = world.get_object_at(gs.location, None, find_held_objects = False)

            if mergeable(agent.holding, obj):
                world.remove(obj)
                o = gs.release() # agent is holding object
                world.remove(agent.holding)
                agent.acquire(obj)
                world.insert(agent.holding)
                # if playable version, merge onto counter first
                # if world.arglist.play:
                #     gs.acquire(agent.holding)
                #     agent.release()

        # if holding something, empty gridsquare in front --> chop or drop
        elif not world.is_occupied(gs.location):
            obj = agent.holding
            if isinstance(gs, Cutboard) and obj.needs_chopped() and not world.arglist.play:
                # normally chop, but if in playable game mode then put down first
                obj.chop()
            else:
                gs.acquire(obj) # obj is put onto gridsquare
                agent.release()
                assert world.get_object_at(gs.location, obj, find_held_objects =\
                    False).is_held == False, "Verifying put down works"

    # if not holding anything
    elif agent.holding is None:
        # not empty in front --> pick up
        if world.is_occupied(gs.location) and not isinstance(gs, Delivery):
            obj = world.get_object_at(gs.location, None, find_held_objects = False)
            # if in playable game mode, then chop raw items on cutting board
            if isinstance(gs, Cutboard) and obj.needs_chopped() and world.arglist.play:
                obj.chop()
            else:
                gs.release()
                agent.acquire(obj)
            obj.last_held = agent.get_team()

        # if empty in front --> interact
        elif not world.is_occupied(gs.location):
            pass
