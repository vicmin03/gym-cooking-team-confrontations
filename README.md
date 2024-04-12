# Code for undergraduate project 'Overcooked!: Developing diverse cooking tasks for multi-agent cooperation and team confrontations


Contents:
- [Introduction](#introduction)
- [Installation](#installation)
- [Running experiments](#experiments)

## Introduction

TODO:
<p align="center">
    <img src="images/2_open_salad.gif" width=260></img>
    <img src="images/2_partial_tl.gif" width=260></img>
    <img src="images/2_full_salad.gif" width=260></img>
</p>


This project is based on the project Too many cooks: Bayesian inference for coordinating by Wang, Rose et al., and extends it by introducing new team features, allowing for two teams of agents to compete in a kitchen environment, navigating to complete dishes and earn a higher score. It also introduces a new method for policy making, Multi-Agent Deep Q Learning, whereby agents are trained to estimate the value of taking actions in each given state of the environment. This should then guide agents to take actions that progress towards completing and delivering dishes for their respective teams, improving their score.
```

## Installation

It is recommended to first create a virtual environment using virtualenv or conda. 

You can install the dependencies with the following commands:
```
cd gym-cooking-team-confrontations
pip3 install -e .
```

## Running experimnts

In order to run experiments, you must be in the directory **gym-cooking-team-confrontations/gym_cooking/** and use Python 3

<p align="center">
    <img src="images/2_open.png" width=260></img>
    <img src="images/3_partial.png" width=260></img>
    <img src="images/4_full.png" width=260></img>
</p>

### Running an experiment 

The basic structure of commands builds on the structure from the original project, so to run agents within a map use the command:

`python main.py --num-agents <number> --level <level name> --model1 <model name> --model2 <model name> --model3 <model name> --model4 <model name>`

where `<number>` is the number of agents interacting in the environment (there can be up to 8 agents, but there is a different limit for each map), 
`<level name>` are the names of levels available under the directory `cooking/utils/levels`, omitting the `.txt.
`<model name>` are the names of models described in the original Overcooked paper, as well as the additional agent type that runs with MADQN rather than BRTDP. Specifically `<model name>` can be replaced with:
* `bd` to run Bayesian Delegation,
* `up` for Uniform Priors,
* `dc` for Divide & Conquer,
* `fb` for Fixed Beliefs,
* `greedy` for Greedy
* 'madqn' for Multi-Agent DQN agents


To train agents using MADQN with different parameters add the following parameters
'train' indicates whether the training process for madqn should be run again
'batch-size' specifies the number of samples used per batch during training
'training-steps' specifies the number of training iterations to be run

For example:
`python main.py --num-agents 2 --level small-map_tomato_teams --model1 madqn--model2 madqn --train --batch-size 512 --training-steps 2000`
will run the MADQN training process over 2000 iterations for two agents in the given map. The updated parameters for the agents' networks will then be saved and can be loaded to play next time with a command such as:
`python main.py --num-agents 2 --level small-map_tomato_teams --model1 madqn --model2 greedy --record`

If madqn agents are run when there are no pre-saved network parameters, the training process will automatically start, using default parameters of batch size 500 and training steps 1000.


Additionally, --record can be used to save screenshots of the game at each time step.
For more information on customisable parameters and their role, use the command:
`python main.py --help'

### Running Experiments

Loss and average reward graphs are automatically generated whenever a new MADQN agent is trained. 
These can be found in the folder gym-cooking/graphs and are named corresponding to the level and number of agents being trained.
