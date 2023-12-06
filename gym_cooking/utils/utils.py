
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

