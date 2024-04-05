def agent_settings(arglist, agent_name):
    if agent_name[-1] == "1": return arglist.model1
    elif agent_name[-1] == "2": return arglist.model2
    elif agent_name[-1] == "3": return arglist.model3
    elif agent_name[-1] == "4": return arglist.model4
    elif agent_name[-1] == "5": return arglist.model5
    elif agent_name[-1] == "6": return arglist.model6
    elif agent_name[-1] == "7": return arglist.model7
    elif agent_name[-1] == "8": return arglist.model8
    else: raise ValueError("Agent name doesn't follow the right naming, `agent-<int>`")

def rep_to_int(rep: str):
    # returns an integer according to the string representing an object
    if rep == 'a': return -1            # agent 
    elif rep == ' ': return 0           # floor (empty)
    elif rep == '-': return 1           # counter
    elif rep == '/': return 2           # cutboard
    elif rep == 't': return 3           # tomato
    elif rep == 'l': return 4           # lettuce
    elif rep == 'o': return 5           # onion
    elif rep == 'p': return 6           # plate
    elif rep == '£': return 7           # delivery blue
    elif rep == "$": return 8           # delivery red
    elif rep == '#': return 9           # trashcan
    elif rep == '*': return 10          # standard delivery (not used)
    elif rep == 't-l': return 11        # merged tomato-lettuce
    elif rep == 'p-l': return 12        # merged plate-lettuce
    elif rep == 'p-t': return 13        # merged plate-tomato
    elif rep == 'S' : return 14         # storage space (for hoarding)

def subtask_to_int(subtask: str):
    # returns an integer according to the subtask 
    if subtask.find('Chop') >= 0: return 1
    elif subtask.find('Merge') >= 0: return 2
    elif subtask.find('Deliver') >= 0: return 3
    elif subtask.find('Hoard') >= 0: return 4
    elif subtask.find('Steal') >= 0: return 5
    elif subtask.find('Trash') >= 0: return 6
    else: return 0              # if no subtask